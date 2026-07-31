#!/bin/bash
#SBATCH --job-name=webb_ulf_preproc
#SBATCH --array=1-65                     # 65 subjects: sub-0001..sub-0065 -> one task each
#SBATCH --cpus-per-task=16               # matches -j 16 in antsMultivariateTemplateConstruction2.sh
#SBATCH --mem=32G
#SBATCH --time=00-006:00:00
#SBATCH --output=logs/webb_ulf_%A_%a.out
#SBATCH --error=logs/webb_ulf_%A_%a.err
#SBATCH --partition=cpu_short

############################   Environment   ############################
module add mrtrix3/3.0 ants/2.4.2 fsl/6.0.1 freesurfer/7.4.1

export FSLOUTPUTTYPE=NIFTI_GZ
# export FS_LICENSE=/path/to/license.txt       # <-- set this if not already
# source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

# --- Fix for: "fast: error while loading shared libraries: libopenblas.so.0" ---
if ! ldd "$(command -v fast)" 2>/dev/null | grep -q "libopenblas.*=>.*/"; then
    module add openblas/dynamic/0.3.7 2>/dev/null || true
    # export LD_LIBRARY_PATH=/gpfs/.../openblas/0.3.7/lib:$LD_LIBRARY_PATH
fi

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs

############################   Paths   ############################
destDir=/gpfs/data/johnsonplab/data/hyperfine/data/ULF_Data/webb
srcRoot=/gpfs/data/johnsonplab/data/hyperfine/data/ULF_Datasets/Webb_data/Data

############################   Subject Selection   ############################
# Map array task ID (1..65) to a zero-padded, two-digit subject index.
i=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

subDir=sub-00${i}     # e.g. sub-0001
subjdir=sub-3${i}      # e.g. sub-301

echo "=============================================="
echo "SLURM task ${SLURM_ARRAY_TASK_ID}  ->  Subject $subDir  =>  $subjdir"
echo "Host: $(hostname)   Started: $(date)"
echo "=============================================="

# refDir depends on subjdir; define here so ULF steps can reference it too.
refDir=$destDir/derivatives/${subjdir}/reference

############################   Clinical Reference Scan   ############################
cDir=$srcRoot/3T/$subDir
if [ -e "$cDir" ]; then
    if [ -e "$destDir/$subjdir/ses-c01/anat/${subjdir}_ses-c01_acq_HF_T1w.nii.gz" ] &&
       [ -e "$destDir/$subjdir/ses-c01/anat/${subjdir}_ses-c01_acq_HF_T2w.nii.gz" ] &&
       [ -e "$destDir/$subjdir/ses-c01/anat/${subjdir}_ses-c01_acq_HF_FLAIR.nii.gz" ]; then

        echo "HF Data Already Processed! Skipping..."
    else
        mkdir -p "$refDir"

        cp "$cDir/anat/${subDir}_acq-highres_T1w.nii.gz"   "$refDir/${subDir}_HF_T1w.nii.gz"
        cp "$cDir/anat/${subDir}_acq-highres_T2w.nii.gz"   "$refDir/${subDir}_HF_T2w.nii.gz"
        cp "$cDir/anat/${subDir}_acq-highres_FLAIR.nii.gz" "$refDir/${subDir}_HF_FLAIR.nii.gz"

        # Bias Correction
        fast -t 1 -B "$refDir/${subDir}_HF_T1w.nii.gz"   -v "$refDir/${subDir}_HF_T1w.nii.gz"
        fast -t 1 -B "$refDir/${subDir}_HF_T2w.nii.gz"   -v "$refDir/${subDir}_HF_T2w.nii.gz"
        fast -t 1 -B "$refDir/${subDir}_HF_FLAIR.nii.gz" -v "$refDir/${subDir}_HF_FLAIR.nii.gz"

        # Processing Files
        mkdir -p "$refDir/processing/fast"
        mv "$refDir"/*.nii* "$refDir/processing/fast" 2>/dev/null || true

        # Resampling
        mkdir -p "$refDir/processing/synthseg" "$refDir/processing/resample"
        mri_synthseg --i "$refDir"/processing/fast/*T1w_restore.nii.gz \
                     --o "$refDir/processing/synthseg/${subDir}_HF_T1w_synthseg.nii.gz" \
                     --resample "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz"
        mri_synthseg --i "$refDir"/processing/fast/*T2w_restore.nii.gz \
                     --o "$refDir/processing/synthseg/${subDir}_HF_T2w_synthseg.nii.gz" \
                     --resample "$refDir/processing/resample/${subDir}_HF_T2w.nii.gz"
        mri_synthseg --i "$refDir"/processing/fast/*FLAIR_restore.nii.gz \
                     --o "$refDir/processing/synthseg/${subDir}_HF_FLAIR_synthseg.nii.gz" \
                     --resample "$refDir/processing/resample/${subDir}_HF_FLAIR.nii.gz"

        # Create Resampled Reference if Doesn't Exist
        [ -e "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz" ]   || cp "$refDir"/processing/fast/*T1w_restore.nii.gz   "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz"
        [ -e "$refDir/processing/resample/${subDir}_HF_T2w.nii.gz" ]   || cp "$refDir"/processing/fast/*T2w_restore.nii.gz   "$refDir/processing/resample/${subDir}_HF_T2w.nii.gz"
        [ -e "$refDir/processing/resample/${subDir}_HF_FLAIR.nii.gz" ] || cp "$refDir"/processing/fast/*FLAIR_restore.nii.gz "$refDir/processing/resample/${subDir}_HF_FLAIR.nii.gz"

        # Registration to T1w
        mkdir -p "$refDir/processing/coregistration"
        flirt -in "$refDir/processing/resample/${subDir}_HF_T2w.nii.gz" \
              -ref "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz" \
              -out "$refDir/processing/coregistration/${subDir}_HF_T2w.nii.gz" \
              -omat "$refDir/processing/coregistration/T2_2_T1.mat" \
              -dof 6 -cost mutualinfo -interp trilinear
        transformconvert "$refDir/processing/coregistration/T2_2_T1.mat" \
            "$refDir/processing/resample/${subDir}_HF_T2w.nii.gz" \
            "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz" \
            flirt_import "$refDir/processing/coregistration/T2_2_T1.txt" -force

        flirt -in "$refDir/processing/resample/${subDir}_HF_FLAIR.nii.gz" \
              -ref "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz" \
              -out "$refDir/processing/coregistration/${subDir}_HF_FLAIR.nii.gz" \
              -omat "$refDir/processing/coregistration/FLAIR_2_T1.mat" \
              -dof 6 -cost mutualinfo -interp trilinear
        transformconvert "$refDir/processing/coregistration/FLAIR_2_T1.mat" \
            "$refDir/processing/resample/${subDir}_HF_FLAIR.nii.gz" \
            "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz" \
            flirt_import "$refDir/processing/coregistration/FLAIR_2_T1.txt" -force

        # Apply Transforms
        mrtransform -linear "$refDir/processing/coregistration/T2_2_T1.txt" \
            -template "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz" \
            "$refDir/processing/resample/${subDir}_HF_T2w.nii.gz" \
            "$refDir/processing/coregistration/${subDir}_HF_T2w.nii.gz" -force
        mrtransform -linear "$refDir/processing/coregistration/FLAIR_2_T1.txt" \
            -template "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz" \
            "$refDir/processing/resample/${subDir}_HF_FLAIR.nii.gz" \
            "$refDir/processing/coregistration/${subDir}_HF_FLAIR.nii.gz" -force

        # Create Directory for Clinical (fixed: use $destDir, not hardcoded /mnt/r/...)
        clinDir=$destDir/$subjdir/ses-c01/anat
        mkdir -p "$clinDir"
        cp "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz"       "$clinDir/${subjdir}_ses-c01_acq_HF_T1w.nii.gz"
        cp "$refDir/processing/coregistration/${subDir}_HF_T2w.nii.gz" "$clinDir/${subjdir}_ses-c01_acq_HF_T2w.nii.gz"
        cp "$refDir/processing/coregistration/${subDir}_HF_FLAIR.nii.gz" "$clinDir/${subjdir}_ses-c01_acq_HF_FLAIR.nii.gz"

        # Create Reference Images
        cp "$refDir/processing/resample/${subDir}_HF_T1w.nii.gz"        "$refDir/T1w_reference.nii.gz"
        cp "$refDir/processing/coregistration/${subDir}_HF_T2w.nii.gz"  "$refDir/T2w_reference.nii.gz"
        cp "$refDir/processing/coregistration/${subDir}_HF_FLAIR.nii.gz" "$refDir/FLAIR_reference.nii.gz"
    fi
fi

############################   Loop ULF Timepoints   ############################
for j in 01 02; do
    sessionDir=ses-${j}
    startdir=$srcRoot/64mT/$subDir/$sessionDir
    drvDir=$destDir/derivatives/$subjdir/$sessionDir
    workdir=$destDir/$subjdir/$sessionDir

    if [ -e "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T1w.nii.gz" ] &&
       [ -e "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T2w.nii.gz" ] &&
       [ -e "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_FLAIR.nii.gz" ]; then
        echo "Subject ${subjdir}_${sessionDir} Data Already Processed! Skipping..."
    else
        if [ -d "$startdir" ]; then
            mkdir -p "$drvDir" "$workdir"

            ##############     Step 1: Gather NII Files     #################
            mkdir -p "$drvDir/raw/nii/T1" "$drvDir/raw/nii/T2" "$drvDir/raw/nii/FLAIR"
            cp "$startdir"/anat/*_T1w.nii.gz   "$drvDir/raw/nii/T1/T1w_axi.nii.gz"
            cp "$startdir"/anat/*_T2w.nii.gz   "$drvDir/raw/nii/T2/T2w_axi.nii.gz"
            cp "$startdir"/anat/*_FLAIR.nii.gz "$drvDir/raw/nii/FLAIR/FLAIR_axi.nii.gz"

            ###############      Step 2: Intensity Correction      ###############
            mkdir -p "$drvDir/processing/bias_correction/T1" \
                     "$drvDir/processing/bias_correction/T2" \
                     "$drvDir/processing/bias_correction/FLAIR"
            fast -t 1 -B "$drvDir/processing/bias_correction/T1/T1w_axi.nii.gz"       "$drvDir/raw/nii/T1/T1w_axi.nii.gz"
            fast -t 1 -B "$drvDir/processing/bias_correction/T2/T2w_axi.nii.gz"       "$drvDir/raw/nii/T2/T2w_axi.nii.gz"
            fast -t 1 -B "$drvDir/processing/bias_correction/FLAIR/FLAIR_axi.nii.gz"  "$drvDir/raw/nii/FLAIR/FLAIR_axi.nii.gz"

            [ -e "$drvDir/processing/bias_correction/T1/T1w_axi_restore.nii.gz" ]      || mv "$drvDir"/raw/nii/T1/T1w_axi_*.nii.gz       "$drvDir/processing/bias_correction/T1/"
            [ -e "$drvDir/processing/bias_correction/T2/T2w_axi_restore.nii.gz" ]      || mv "$drvDir"/raw/nii/T2/T2w_axi_*.nii.gz       "$drvDir/processing/bias_correction/T2/"
            [ -e "$drvDir/processing/bias_correction/FLAIR/FLAIR_axi_restore.nii.gz" ] || mv "$drvDir"/raw/nii/FLAIR/FLAIR_axi_*.nii.gz  "$drvDir/processing/bias_correction/FLAIR/"

            ###############          Step 3: ULF Resampling         ###############
            mkdir -p "$drvDir/processing/synthseg" "$drvDir/processing/resample"
            mri_synthseg --i "$drvDir/processing/bias_correction/T1/T1w_axi_restore.nii.gz" \
                        --o "$drvDir/processing/synthseg/T1_axi_synthseg.nii.gz" \
                        --resample "$drvDir/processing/resample/T1_axi.nii.gz"
            mri_synthseg --i "$drvDir/processing/bias_correction/T2/T2w_axi_restore.nii.gz" \
                        --o "$drvDir/processing/synthseg/T2_axi_synthseg.nii.gz" \
                        --resample "$drvDir/processing/resample/T2_axi.nii.gz"
            mri_synthseg --i "$drvDir/processing/bias_correction/FLAIR/FLAIR_axi_restore.nii.gz" \
                        --o "$drvDir/processing/synthseg/FLAIR_axi_synthseg.nii.gz" \
                        --resample "$drvDir/processing/resample/FLAIR_axi.nii.gz"

            ###############     Step 4: Longitudinal Resampling & Registration    ###############
            mkdir -p "$drvDir/processing/coregistration/T1/transforms" \
                     "$drvDir/processing/coregistration/T2/transforms" \
                     "$drvDir/processing/coregistration/FLAIR/transforms"

            if [ "$j" = "01" ]; then
                if [ -e "$refDir/T2w_reference.nii.gz" ]; then
                    flirt -in "$drvDir/processing/resample/T2_axi.nii.gz" \
                        -ref "$refDir/T2w_reference.nii.gz" \
                        -out "$drvDir/processing/coregistration/T2_axi.nii.gz" \
                        -omat "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                        -dof 6 -cost mutualinfo -interp trilinear
                    transformconvert "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                        "$drvDir/processing/resample/T2_axi.nii.gz" \
                        "$refDir/T2w_reference.nii.gz" flirt_import \
                        "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" -force
                    mrtransform -linear "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" \
                        -template "$refDir/T2w_reference.nii.gz" \
                        "$drvDir/processing/resample/T2_axi.nii.gz" \
                        "$drvDir/processing/coregistration/T2_axi.nii.gz" -force
                else
                    echo "CLINICAL REFERENCE MISSING!"
                    cp "$drvDir/processing/resample/T2_axi.nii.gz" "$drvDir/processing/coregistration/T2_axi.nii.gz"
                fi
            else
                # Fixed: use $destDir instead of hardcoded /mnt/r/Peter/...
                t1dir=$destDir/${subjdir}/ses-01
                if [ -e "${t1dir}/anat/${subjdir}_ses-01_acq-ULF_Axial_T2w.nii.gz" ]; then
                    echo "Registration to Timepoint 1"
                    flirt -in "$drvDir/processing/resample/T2_axi.nii.gz" \
                        -ref "${t1dir}/anat/${subjdir}_ses-01_acq-ULF_Axial_T2w.nii.gz" \
                        -out "$drvDir/processing/coregistration/T2_axi.nii.gz" \
                        -omat "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                        -dof 6 -cost mutualinfo -interp trilinear
                    transformconvert "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.mat" \
                        "$drvDir/processing/resample/T2_axi.nii.gz" \
                        "${t1dir}/anat/${subjdir}_ses-01_acq-ULF_Axial_T2w.nii.gz" flirt_import \
                        "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" -force
                    mrtransform -linear "$drvDir/processing/coregistration/T2/transforms/T2_ULF2REF.txt" \
                        -template "${t1dir}/anat/${subjdir}_ses-01_acq-ULF_Axial_T2w.nii.gz" \
                        "$drvDir/processing/resample/T2_axi.nii.gz" \
                        "$drvDir/processing/coregistration/T2_axi.nii.gz" -force
                else
                    echo "Missing ses-01 reference template for $subjdir $sessionDir"
                fi
            fi

            #################      Step 6: Register T1 & FLAIR to T2 (within session)      ##################
            if [ -e "$drvDir/processing/coregistration/T2_axi.nii.gz" ]; then
                flirt -in "$drvDir/processing/resample/T1_axi.nii.gz" \
                      -ref "$drvDir/processing/coregistration/T2_axi.nii.gz" \
                      -out "$drvDir/processing/coregistration/T1_axi.nii.gz" \
                      -omat "$drvDir/processing/coregistration/T1/transforms/T1_to_T2.mat" \
                      -dof 6 -cost mutualinfo -interp trilinear
            fi
            transformconvert "$drvDir/processing/coregistration/T1/transforms/T1_to_T2.mat" \
                "$drvDir/processing/resample/T1_axi.nii.gz" \
                "$drvDir/processing/coregistration/T2_axi.nii.gz" flirt_import \
                "$drvDir/processing/coregistration/T1/transforms/T1_to_T2.txt" -force
            mrtransform -linear "$drvDir/processing/coregistration/T1/transforms/T1_to_T2.txt" \
                -template "$drvDir/processing/coregistration/T2_axi.nii.gz" \
                "$drvDir/processing/resample/T1_axi.nii.gz" \
                "$drvDir/processing/coregistration/T1_axi.nii.gz" -force

            if [ -e "$drvDir/processing/coregistration/T2_axi.nii.gz" ]; then
                flirt -in "$drvDir/processing/resample/FLAIR_axi.nii.gz" \
                      -ref "$drvDir/processing/coregistration/T2_axi.nii.gz" \
                      -out "$drvDir/processing/coregistration/FLAIR_axi.nii.gz" \
                      -omat "$drvDir/processing/coregistration/FLAIR/transforms/FLAIR_to_T2.mat" \
                      -dof 6 -cost mutualinfo -interp trilinear
            fi
            transformconvert "$drvDir/processing/coregistration/FLAIR/transforms/FLAIR_to_T2.mat" \
                "$drvDir/processing/resample/FLAIR_axi.nii.gz" \
                "$drvDir/processing/coregistration/T2_axi.nii.gz" flirt_import \
                "$drvDir/processing/coregistration/FLAIR/transforms/FLAIR_to_T2.txt" -force
            mrtransform -linear "$drvDir/processing/coregistration/FLAIR/transforms/FLAIR_to_T2.txt" \
                -template "$drvDir/processing/coregistration/T2_axi.nii.gz" \
                "$drvDir/processing/resample/FLAIR_axi.nii.gz" \
                "$drvDir/processing/coregistration/FLAIR_axi.nii.gz" -force

            ##############        Step 8: Move Processed Data        ######################
            mkdir -p "$workdir/anat"
            cp "$drvDir/processing/coregistration/T1_axi.nii.gz"    "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T1w.nii.gz"
            cp "$drvDir/processing/coregistration/T2_axi.nii.gz"    "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_T2w.nii.gz"
            cp "$drvDir/processing/coregistration/FLAIR_axi.nii.gz" "$workdir/anat/${subjdir}_${sessionDir}_acq-ULF_Axial_FLAIR.nii.gz"
        else
            echo "$startdir Does Not Exist"
        fi
    fi
done

echo "Finished subject $subjdir at $(date)"