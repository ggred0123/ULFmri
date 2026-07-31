#!/bin/bash
#SBATCH --job-name=kcl_ulf_preproc
#SBATCH --array=0-22                     # 23 subjects: HYPE00..HYPE22
#SBATCH --cpus-per-task=16               # matches -j 16 in antsMultivariateTemplateConstruction2.sh
#SBATCH --mem=32G
#SBATCH --time=00-012:00:00
#SBATCH --output=logs/kcl_ulf_%A_%a.out
#SBATCH --error=logs/kcl_ulf_%A_%a.err
#SBATCH --partition=cpu_short


############################   Environment   ############################
module add mrtrix3/3.0 ants/2.4.2 fsl/6.0.1 freesurfer/7.4.1

# --- FSL / FreeSurfer setup (adjust paths if your cluster differs) ---
# Ensure FSL runtime variables are set (some modules do this automatically).
export FSLOUTPUTTYPE=NIFTI_GZ
# FreeSurfer needs a license and SUBJECTS_DIR for SynthSeg / mri_synthseg.
# export FREESURFER_HOME=/path/to/freesurfer   # usually set by the module
# export FS_LICENSE=/path/to/license.txt       # <-- set this if not already
# source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

# --- Fix for: "fast: error while loading shared libraries: libopenblas.so.0" ---
# If your fsl/6.0.0 module does not resolve libopenblas.so.0, point the linker
# at an OpenBLAS lib directory (adjust to your cluster's actual path).
# Verify with:  ldd $(which fast) | grep openblas
if ! ldd "$(command -v fast)" 2>/dev/null | grep -q "libopenblas.*=>.*/"; then
    module add openblas/dynamic/0.3.7 2>/dev/null || true
    # As a fallback, uncomment and set the correct path:
    # export LD_LIBRARY_PATH=/gpfs/.../openblas/0.3.7/lib:$LD_LIBRARY_PATH
fi

# Threading hints for ANTs / FSL
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs

############################   Paths   ############################
destDir=/gpfs/data/johnsonplab/data/hyperfine/data/ULF_Data/kcl

############################   Subject Selection   ############################
# Convert the array task ID (0..22) into a zero-padded, two-digit index.
i=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

subDir=sub-HYPE${i}
subjdir=sub-2${i}

echo "=============================================="
echo "SLURM task ${SLURM_ARRAY_TASK_ID}  ->  Subject $subDir  =>  $subjdir"
echo "Host: $(hostname)   Started: $(date)"
echo "=============================================="

############################   Clinical Reference Scan   ############################
cDir=/gpfs/data/johnsonplab/data/hyperfine/data/ULF_Datasets/ULF_KCL/$subDir/ses-GE
refDir=$destDir/derivatives/${subjdir}/reference
if [ -e "$cDir" ]; then
    if [ -e "$destDir/$subjdir/ses-c01/anat/${subjdir}_ses-c01_acq_HF_T1w.nii.gz" ] &&
       [ -e "$destDir/$subjdir/ses-c01/anat/${subjdir}_ses-c01_acq_HF_T2w.nii.gz" ]; then

        echo "HF Data Already Processed! Skipping..."

    else
        mkdir -p "$refDir"
        cp "$cDir/anat/${subDir}_ses-GE_acq-MPRAGE_T1w.nii.gz" \
           "$refDir/${subDir}_ses-GE_acq-MPRAGE_T1w.nii.gz"
        cp "$cDir/anat/${subDir}_ses-GE_T2w.nii.gz" \
           "$refDir/${subDir}_ses-GE_T2w.nii.gz"

        # Bias Correction
        fast -t 1 -B "$refDir/${subDir}_ses-GE_acq-MPRAGE_T1w.nii.gz" \
                 -v "$refDir/${subDir}_ses-GE_acq-MPRAGE_T1w.nii.gz"
        fast -t 1 -B "$refDir/${subDir}_ses-GE_T2w.nii.gz" \
                 -v "$refDir/${subDir}_ses-GE_T2w.nii.gz"

        # Processing Files (mkdir -p makes this idempotent on re-runs)
        mkdir -p "$refDir/processing"
        mv "$refDir"/*.nii* "$refDir/processing"/ 2>/dev/null || true

        # Guard the copies: only copy if fast actually produced the *_restore files
        shopt -s nullglob
        t1_restore=("$refDir"/processing/*T1w_restore.nii.gz)
        t2_restore=("$refDir"/processing/*T2w_restore.nii.gz)
        if (( ${#t1_restore[@]} )) && (( ${#t2_restore[@]} )); then
            cp "${t1_restore[0]}" "$refDir/${subDir}_ses-GE_T1w.nii.gz"
            cp "${t2_restore[0]}" "$refDir/${subDir}_ses-GE_T2w.nii.gz"
        else
            echo "ERROR: fast did not produce *_restore outputs for $subDir. Check OpenBLAS/libs." >&2
            exit 1
        fi
        shopt -u nullglob

        # Resampling
        mri_synthseg --i "$refDir/${subDir}_ses-GE_T1w.nii.gz" \
                     --o "$refDir/${subDir}_ses-GE_T1w_synthseg.nii.gz" \
                     --resample "$refDir/T1w_reference.nii.gz"
        mri_synthseg --i "$refDir/${subDir}_ses-GE_T2w.nii.gz" \
                     --o "$refDir/${subDir}_ses-GE_T2w_synthseg.nii.gz" \
                     --resample "$refDir/T2w_reference.nii.gz"

        # Create Reference if Doesn't Exist
        if [ -e "$refDir/T1w_reference.nii.gz" ]; then
            echo "reference image exists"
        else
            cp "$refDir/${subDir}_ses-GE_T1w.nii.gz" "$refDir/T1w_reference.nii.gz"
        fi
        if [ -e "$refDir/T2w_reference.nii.gz" ]; then
            echo "reference image exists"
        else
            cp "$refDir/${subDir}_ses-GE_T2w.nii.gz" "$refDir/T2w_reference.nii.gz"
        fi

        # Coregistration T1 to T2
        flirt -in "$refDir/T1w_reference.nii.gz" \
              -ref "$refDir/T2w_reference.nii.gz" \
              -out "$refDir/T1w_reference_coreg.nii.gz" \
              -omat "$refDir/T1_2_T2.mat" \
              -dof 6 -cost mutualinfo -interp trilinear

        transformconvert "$refDir/T1_2_T2.mat" \
            "$refDir/T1w_reference.nii.gz" \
            "$refDir/T2w_reference.nii.gz" \
            flirt_import "$refDir/T1_2_T2.txt" -force

        mrtransform -linear "$refDir/T1_2_T2.txt" \
            -template "$refDir/T2w_reference.nii.gz" \
            "$refDir/T1w_reference.nii.gz" \
            "$refDir/T1w_reference_coreg.nii.gz" -force

        # Create Directory for Clinical
        clinDir=$destDir/$subjdir/ses-c01/anat
        mkdir -p "$clinDir"
        cp "$refDir/T1w_reference_coreg.nii.gz" "$clinDir/${subjdir}_ses-c01_acq_HF_T1w.nii.gz"
        cp "$refDir/T2w_reference.nii.gz" "$clinDir/${subjdir}_ses-c01_acq_HF_T2w.nii.gz"
    fi
fi

############################   Loop ULF Timepoints   ############################
for k in HFC HFE; do

    case $k in
        HFC) j="001" ;;
        HFE) j="002" ;;
    esac

    sesDir=ses-${k}
    sessionDir=ses-${j}

    startdir=/gpfs/data/johnsonplab/data/hyperfine/data/ULF_Datasets/ULF_KCL/$subDir/$sesDir
    drvDir=$destDir/derivatives/$subjdir/$sessionDir
    workdir=$destDir/$subjdir/$sessionDir

    if [ -e "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T1w.nii.gz" ] &&
       [ -e "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T2w.nii.gz" ]; then

        echo "${subjdir}_${sessionDir} Already Preprocessed! Skipping..."
        continue
    fi

    mkdir -p "$drvDir" "$workdir"

    ##############     Step 1: NII Files     #################
    mkdir -p "$drvDir/raw/nii/T1" "$drvDir/raw/nii/T2"

    cp "$startdir"/anat/*axi_T1w.nii.gz "$drvDir/raw/nii/T1/T1w_axi.nii.gz"
    cp "$startdir"/anat/*cor_T1w.nii.gz "$drvDir/raw/nii/T1/T1w_cor.nii.gz"
    cp "$startdir"/anat/*sag_T1w.nii.gz "$drvDir/raw/nii/T1/T1w_sag.nii.gz"
    cp "$startdir"/anat/*axi_T2w.nii.gz "$drvDir/raw/nii/T2/T2w_axi.nii.gz"
    cp "$startdir"/anat/*cor_T2w.nii.gz "$drvDir/raw/nii/T2/T2w_cor.nii.gz"
    cp "$startdir"/anat/*sag_T2w.nii.gz "$drvDir/raw/nii/T2/T2w_sag.nii.gz"

    ###############      Step 2: Intensity Correction      ###############
    mkdir -p "$drvDir/processing/bias_correction/T1" "$drvDir/processing/bias_correction/T2"

    fast -t 1 -B "$drvDir/processing/bias_correction/T1/T1w_axi.nii.gz" "$drvDir/raw/nii/T1/T1w_axi.nii.gz"
    fast -t 1 -B "$drvDir/processing/bias_correction/T1/T1w_cor.nii.gz" "$drvDir/raw/nii/T1/T1w_cor.nii.gz"
    fast -t 1 -B "$drvDir/processing/bias_correction/T1/T1w_sag.nii.gz" "$drvDir/raw/nii/T1/T1w_sag.nii.gz"

    fast -t 2 -B "$drvDir/processing/bias_correction/T2/T2w_axi.nii.gz" "$drvDir/raw/nii/T2/T2w_axi.nii.gz"
    fast -t 2 -B "$drvDir/processing/bias_correction/T2/T2w_cor.nii.gz" "$drvDir/raw/nii/T2/T2w_cor.nii.gz"
    fast -t 2 -B "$drvDir/processing/bias_correction/T2/T2w_sag.nii.gz" "$drvDir/raw/nii/T2/T2w_sag.nii.gz"

    for con in T1 T2; do
        for ori in axi cor sag; do
            if [ -e "$drvDir/processing/bias_correction/${con}/${con}w_${ori}_restore.nii.gz" ]; then
                echo "fast worked for ${con} ${ori}"
            else
                mv $drvDir/raw/nii/${con}/${con}w_${ori}_*.nii.gz $drvDir/processing/bias_correction/${con}
            fi
        done
    done

    ###############     Step 3: ANTS Multivariable Template     ###############
    mkdir -p "$drvDir/processing/antsTemplate/T2"
    cd "$drvDir/processing/antsTemplate/T2"

    antsMultivariateTemplateConstruction2.sh -d 3 -a 2 -o T2_ -c 2 -j "${SLURM_CPUS_PER_TASK:-16}" \
        -n 1 -r 1 -l 0 -m MI -t Affine "$drvDir"/processing/bias_correction/T2/T2*restore*

    mrconvert -strides 1,2,3 "$drvDir/processing/antsTemplate/T2/T2_template0.nii.gz" \
        "$drvDir/processing/antsTemplate/T2/T2_template0.nii.gz" -force

    #rm -rf intermediateTemplates
    #rm -f "$drvDir"/processing/antsTemplate/T2/*Warp*
    #rm -f "$drvDir"/processing/antsTemplate/T2/*Repair*

    mkdir -p "$drvDir/processing/antsTemplate/T1"
    cd "$drvDir/processing/antsTemplate/T1"

    antsMultivariateTemplateConstruction2.sh -d 3 -a 2 -o T1_ -c 2 -j "${SLURM_CPUS_PER_TASK:-16}" \
        -n 1 -r 1 -l 0 -m MI -t Affine "$drvDir"/processing/bias_correction/T1/T1*restore*

    mrconvert -strides 1,2,3 "$drvDir/processing/antsTemplate/T1/T1_template0.nii.gz" \
        "$drvDir/processing/antsTemplate/T1/T1_template0.nii.gz" -force

    #rm -rf intermediateTemplates
    #rm -f "$drvDir"/processing/antsTemplate/T1/*Warp*
    #rm -f "$drvDir"/processing/antsTemplate/T1/*Repair*

    mkdir -p "$drvDir/processing/coregistration/T1/transforms" \
             "$drvDir/processing/coregistration/T2/transforms"
    mv "$drvDir"/processing/antsTemplate/T1/*.mat "$drvDir/processing/coregistration/T1/transforms"
    mv "$drvDir"/processing/antsTemplate/T2/*.mat "$drvDir/processing/coregistration/T2/transforms"

    ###############          Step 4: ULF Resampling         ###############
    mkdir -p "$drvDir/processing/synthseg" "$drvDir/processing/resample"

    mri_synthseg --i "$drvDir/processing/antsTemplate/T1/T1_template0.nii.gz" \
                 --o "$drvDir/processing/synthseg/T1_TomoBrain_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T1_TomoBrain.nii.gz"
    mri_synthseg --i "$drvDir/processing/bias_correction/T1/T1w_axi_restore.nii.gz" \
                 --o "$drvDir/processing/synthseg/T1_axi_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T1_axi.nii.gz"
    mri_synthseg --i "$drvDir/processing/bias_correction/T1/T1w_cor_restore.nii.gz" \
                 --o "$drvDir/processing/synthseg/T1_cor_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T1_cor.nii.gz"
    mri_synthseg --i "$drvDir/processing/bias_correction/T1/T1w_sag_restore.nii.gz" \
                 --o "$drvDir/processing/synthseg/T1_sag_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T1_sag.nii.gz"

    mri_synthseg --i "$drvDir/processing/antsTemplate/T2/T2_template0.nii.gz" \
                 --o "$drvDir/processing/synthseg/T2_TomoBrain_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T2_TomoBrain.nii.gz"
    mri_synthseg --i "$drvDir/processing/bias_correction/T2/T2w_axi_restore.nii.gz" \
                 --o "$drvDir/processing/synthseg/T2_axi_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T2_axi.nii.gz"
    mri_synthseg --i "$drvDir/processing/bias_correction/T2/T2w_cor_restore.nii.gz" \
                 --o "$drvDir/processing/synthseg/T2_cor_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T2_cor.nii.gz"
    mri_synthseg --i "$drvDir/processing/bias_correction/T2/T2w_sag_restore.nii.gz" \
                 --o "$drvDir/processing/synthseg/T2_sag_synthseg.nii.gz" \
                 --resample "$drvDir/processing/resample/T2_sag.nii.gz"

    ###############     Step 5: Longitudinal Resampling & Registration    ###############
    if [ "$j" = "001" ]; then
        if [ -e "$refDir/T2w_reference.nii.gz" ]; then
            flirt -in "$drvDir/processing/resample/T2_TomoBrain.nii.gz" \
                  -ref "$refDir/T2w_reference.nii.gz" \
                  -out "$drvDir/processing/coregistration/T2_TomoBrain.nii.gz" \
                  -omat "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                  -dof 6 -cost mutualinfo -interp trilinear

            transformconvert \
                "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                "$drvDir/processing/resample/T2_TomoBrain.nii.gz" \
                "$refDir/T2w_reference.nii.gz" \
                flirt_import \
                "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" -force
        else
            echo "CLINICAL REFERENCE MISSING!"
        fi
    else
        t1dir=$destDir/$subjdir/ses-001
        if [ -e "$t1dir/anat/${subjdir}_ses-001_acq-ULF_TomoBrain_T2w.nii.gz" ]; then
            flirt -in "$drvDir/processing/resample/T2_TomoBrain.nii.gz" \
                  -ref "$t1dir/anat/${subjdir}_ses-001_acq-ULF_TomoBrain_T2w.nii.gz" \
                  -out "$drvDir/processing/coregistration/T2_TomoBrain.nii.gz" \
                  -omat "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                  -dof 6 -cost mutualinfo -interp trilinear

            transformconvert \
                "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                "$drvDir/processing/resample/T2_TomoBrain.nii.gz" \
                "$t1dir/anat/${subjdir}_ses-001_acq-ULF_TomoBrain_T2w.nii.gz" \
                flirt_import \
                "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" -force
        else
            echo "Missing ses-001 reference template for $subjdir $sessionDir"
        fi
    fi

    #################      Step 6: Registration of T1 to T2 (within session)      ##################
    if [ -e "$drvDir/processing/coregistration/T2_TomoBrain.nii.gz" ]; then
        flirt -in "$drvDir/processing/resample/T1_TomoBrain.nii.gz" \
              -ref "$drvDir/processing/coregistration/T2_TomoBrain.nii.gz" \
              -out "$drvDir/processing/coregistration/T1_TomoBrain.nii.gz" \
              -omat "$drvDir/processing/coregistration/T1/transforms/T1_to_T2.mat" \
              -dof 6 -cost mutualinfo -interp trilinear
    fi

    transformconvert \
        "$drvDir/processing/coregistration/T1/transforms/T1_to_T2.mat" \
        "$drvDir/processing/resample/T1_TomoBrain.nii.gz" \
        "$drvDir/processing/coregistration/T2_TomoBrain.nii.gz" \
        flirt_import \
        "$drvDir/processing/coregistration/T1/transforms/T1_to_T2.txt" -force

    ############   Step 7: Apply Transforms to Orthogonal Directions   #############
    # --- T1 orthogonals ---
    for ori in axi cor sag; do
        # Glob for the actual .mat file matching this orientation
        transdir=$drvDir/processing/coregistration/T1/transforms
        mat_files=("$transdir"/*_${ori}*GenericAffine.mat)

        # Safety check: make sure exactly one match was found
        if (( ${#mat_files[@]} == 0 )); then
            echo "ERROR: no T1 GenericAffine .mat found for $ori in $transdir" >&2
            continue
        elif (( ${#mat_files[@]} > 1 )); then
            echo "WARNING: multiple T1 .mat files match *_${ori}*GenericAffine.mat; using first: ${mat_files[0]}" >&2
        fi

        matfile="${mat_files[0]}"
        # Derive the base name (without .mat) for the .txt outputs
        gen=$(basename "$matfile" .mat)

        ConvertTransformFile 3 \
            "$matfile" \
            "$transdir/${gen}.txt"
        transformconvert \
            "$transdir/${gen}.txt" \
            itk_import \
            "$transdir/${gen}.txt" -force

        transformcompose \
            "$transdir/T1_to_T2.txt" \
            "$transdir/${gen}.txt" \
            "$transdir/T1_${ori}_Affine.txt" -force

        mrtransform \
            -linear "$transdir/T1_${ori}_Affine.txt" \
            -template "$drvDir/processing/coregistration/T1_TomoBrain.nii.gz" \
            "$drvDir/processing/resample/T1_${ori}.nii.gz" \
            "$drvDir/processing/coregistration/T1w_${ori}.nii.gz" -force
    done

    # --- T2 orthogonals ---
    for ori in axi cor sag; do
        # Glob for the actual .mat file matching this orientation
        transdir=$drvDir/processing/coregistration/T2/transforms
        mat_files=("$transdir"/*_${ori}*GenericAffine.mat)

        # Safety check: make sure exactly one match was found
        if (( ${#mat_files[@]} == 0 )); then
            echo "ERROR: no T2 GenericAffine .mat found for $ori in $transdir" >&2
            continue
        elif (( ${#mat_files[@]} > 1 )); then
            echo "WARNING: multiple T2 .mat files match *_${ori}*GenericAffine.mat; using first: ${mat_files[0]}" >&2
        fi

        matfile="${mat_files[0]}"
        # Derive the base name (without .mat) for the .txt outputs
        gen=$(basename "$matfile" .mat)

        ConvertTransformFile 3 \
            "$drvDir/processing/coregistration/T2/transforms/${gen}.mat" \
            "$drvDir/processing/coregistration/T2/transforms/${gen}.txt"
        transformconvert \
            "$drvDir/processing/coregistration/T2/transforms/${gen}.txt" \
            itk_import \
            "$drvDir/processing/coregistration/T2/transforms/${gen}.txt" -force

        if [ -e "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" ]; then
            transformcompose \
                "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" \
                "$drvDir/processing/coregistration/T2/transforms/${gen}.txt" \
                "$drvDir/processing/coregistration/T2/transforms/T2_${ori}_Affine.txt" -force

            mrtransform \
                -linear "$drvDir/processing/coregistration/T2/transforms/T2_${ori}_Affine.txt" \
                -template "$drvDir/processing/coregistration/T2_TomoBrain.nii.gz" \
                "$drvDir/processing/resample/T2_${ori}.nii.gz" \
                "$drvDir/processing/coregistration/T2w_${ori}.nii.gz" -force
        else
            mrtransform -linear \
                "$drvDir/processing/coregistration/T2/transforms/${gen}.txt" \
                "$drvDir/processing/resample/T2_${ori}.nii.gz" \
                "$drvDir/processing/coregistration/T2w_${ori}.nii.gz" -force
        fi
    done

    ##############        Step 8: Move Processed Data (consistent naming)        ######################
    mkdir -p "$workdir/anat"

    cp "$drvDir/processing/coregistration/T1_TomoBrain.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_TomoBrain_T1w.nii.gz"
    cp "$drvDir/processing/coregistration/T1w_axi.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T1w.nii.gz"
    cp "$drvDir/processing/coregistration/T1w_cor.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Coronal_T1w.nii.gz"
    cp "$drvDir/processing/coregistration/T1w_sag.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Sagittal_T1w.nii.gz"

    cp "$drvDir/processing/coregistration/T2_TomoBrain.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_TomoBrain_T2w.nii.gz"
    cp "$drvDir/processing/coregistration/T2w_axi.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T2w.nii.gz"
    cp "$drvDir/processing/coregistration/T2w_cor.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Coronal_T2w.nii.gz"
    cp "$drvDir/processing/coregistration/T2w_sag.nii.gz" \
        "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Sagittal_T2w.nii.gz"

done

echo "Finished subject $subjdir at $(date)"