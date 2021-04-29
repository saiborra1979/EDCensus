# import numpy as np
# import pandas as pd
# columns = ["lab"]
# df = pd.read_csv("u_DI.csv", usecols=columns)
# DIseries = df["lab"]

brain_MRI = ['MRI Consultation Neuro', 'MRI Brain w contrast', 'MRI Brain & Orbits w contrast',
             'MRI Brain & IAC w & w/o contrast', 'MRI Brain & Neck w/o contrast', 'MRI C-T-L Spine w contrast',
             'MRI Neck w/o contrast', 'MRI Lumbar Spine w/o contrast', 'MRI Face w & w/o contrast',
             'MRI Brain & Neck w & w/o contrast', 'MRI Cerv Thoracic w/o contrast', 'MRI Brain w/o contrast',
             'MRI Brain w & w/o contrast', 'MRI C-T-L Spine w/o contrast', 'MRI Cervical Spine w/o contrast',
             'MRI Brain & Orbits w & w/o contrast', 'MRI C-T-L Spine w & w/o contrast',
             'MRI Brain & Orbits w/o contrast']

DI_tags = {
    # For empty values
    'none': ['none'],

    # X-Ray Ankle Labels
    'X-Ray Ankle LEFT 3 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankles BILATERAL 3 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankle RIGHT 3 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankles Bilat 3 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankle Lt 2 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankle Lt 3 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankle Rt 3 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankle RIGHT 2 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankles BILATERAL 2 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankle LEFT 2 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankle Rt 2 Views': ['X-Ray', 'Ankle'],
    'X-Ray Ankles Bilat 2 View': ['X-Ray', 'Ankle'],
    'X-Ray Ankles Bilat 2 Views': ['X-Ray', 'Ankle'],

    # X-Ray Foot Labels
    'X-Ray Foot RIGHT 3 Views': ['X-Ray', 'Foot'],
    'X-Ray Foot LEFT 3 Views': ['X-Ray', 'Foot'],
    'X-Ray Foot RIGHT 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Foot Rt 3 Views': ['X-Ray', 'Foot'],
    'X-Ray Foot Lt 3 Views': ['X-Ray', 'Foot'],
    'X-Ray Foot Lt 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Foot Rt 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Foot LEFT 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Feet BILATERAL 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Knees BILATERAL 3 Views +': ['X-Ray', 'Foot'],
    'X-Ray Feet Bilat 3 Views': ['X-Ray', 'Foot'],
    'X-Ray Feet BILATERAL 3 Views': ['X-Ray', 'Foot'],
    'X-Ray Feet Bilat 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Calcaneus Bilat 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Calcaneus LEFT 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Calcaneus Rt 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Calcaneus RIGHT 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Calcaneus Lt 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Toe RIGHT 3 Views +': ['X-Ray', 'Foot'],
    'X-Ray Toe LEFT 3 Views +': ['X-Ray', 'Foot'],
    'X-Ray Toe Lt 3 Views +': ['X-Ray', 'Foot'],
    'X-Ray Toe Rt 3 Views +': ['X-Ray', 'Foot'],

    # X-Ray Leg Labels
    'X-Ray Tibia & Fibula LEFT 2 Views': ['X-Ray', 'Leg'],
    'X-Ray Tibia & Fibula RIGHT 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Tibia & Fibula Lt 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Tibia & Fibula Rt 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Tibia & Fibula Bilateral 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Tibia & Fibula Bilateral 3 Views +': ['X-Ray', 'Foot'],
    'X-Ray Tibia & Fibula BILATERAL 2 Views': ['X-Ray', 'Foot'],
    'X-Ray Tibia & Fibula Rt 3 Views +': ['X-Ray', 'Foot'],
    "X-Ray 3'/4' Legs 1 View": ['X-Ray', 'Foot'],

    # X-Ray Knee Labels
    'X-Ray Knee RIGHT 2 Views': ['X-Ray', 'Knee'],
    'X-Ray Knee LEFT 2 Views': ['X-Ray', 'Knee'],
    'X-Ray Knee Lt 2 Views': ['X-Ray', 'Knee'],
    'X-Ray Knees Bilat 2 Views': ['X-Ray', 'Knee'],
    'X-Ray Knee Rt 2 Views': ['X-Ray', 'Knee'],
    'X-Ray Knee LEFT 3 Views +': ['X-Ray', 'Knee'],
    'X-Ray Knee Rt 3 Views +': ['X-Ray', 'Knee'],
    'X-Ray Knees BILATERAL 2 Views': ['X-Ray', 'Knee'],
    'X-Ray Knee Lt 3 Views +': ['X-Ray', 'Knee'],
    'X-Ray Knee RIGHT 3 Views +': ['X-Ray', 'Knee'],
    'X-Ray Knees Bilat 3 Views +': ['X-Ray', 'Knee'],
    'X-Ray Patella / Knee LEFT 3+  Views': ['X-Ray', 'Knee'],
    'X-Ray Patella / Knee RIGHT 3+  Views': ['X-Ray', 'Knee'],
    'X-Ray Patella / Knee Rt 3+  Views': ['X-Ray', 'Knee'],
    'X-Ray Patella / Knees BILATERAL 3 Views': ['X-Ray', 'Knee'],
    'X-Ray Patella / Knee Lt 3+  Views': ['X-Ray', 'Knee'],

    # X-Ray Thigh Labels
    'X-Ray Femur LEFT': ['X-Ray', 'Thigh'],
    'X-Ray Femur RIGHT': ['X-Ray', 'Thigh'],
    'X-Ray Femur Lt': ['X-Ray', 'Thigh'],
    'X-Ray Femur Rt': ['X-Ray', 'Thigh'],
    'X-Ray Femur BILATERAL': ['X-Ray', 'Thigh'],
    'X-Ray Femur Bilat': ['X-Ray', 'Thigh'],

    # X-Ray Pelvis Labels
    'X-Ray Pelvis 1 View': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Pelvis 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Pelvis 3 Views +': ['X-Ray', 'Pelvis/Hips'],
    'Xray Pelvis': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Hips Bilat 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Hips BILATERAL 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Hip RIGHT 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Hip LEFT 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Hip Rt 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Hip Lt 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Sacral Iliac Joints Bilat': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Sacral Iliac Joints BILATERAL': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Sacrum/Coccyx 2 Views': ['X-Ray', 'Pelvis/Hips'],
    'X-Ray Sacrum/Coccyx 3 Views +': ['X-Ray', 'Pelvis/Hips'],

    # X-Ray Hand Labels
    'X-Ray Hand LEFT 3 Views': ['X-Ray', 'Hand'],
    'X-Ray Hand RIGHT 3 Views': ['X-Ray', 'Hand'],
    'X-Ray Hand Rt 3 Views': ['X-Ray', 'Hand'],
    'X-Ray Hand Lt 3 Views': ['X-Ray', 'Hand'],
    'X-Ray Hand Bilat 3 Views': ['X-Ray', 'Hand'],
    'X-Ray Hand BILATERAL 3 Views': ['X-Ray', 'Hand'],
    'X-Ray Finger RIGHT': ['X-Ray', 'Hand'],
    'X-Ray Finger LEFT': ['X-Ray', 'Hand'],
    'X-Ray Finger Rt': ['X-Ray', 'Hand'],
    'X-Ray Finger Lt': ['X-Ray', 'Hand'],

    # X-Ray Wrist Labels
    'X-Ray Wrist RIGHT 3 Views +': ['X-Ray', 'Wrist'],
    'X-Ray Wrist LEFT 3 Views +': ['X-Ray', 'Wrist'],
    'X-Ray Wrist RIGHT 2 Views': ['X-Ray', 'Wrist'],
    'X-Ray Wrist Rt 3 Views +': ['X-Ray', 'Wrist'],
    'X-Ray Wrist Lt 3 Views +': ['X-Ray', 'Wrist'],
    'X-Ray Wrist Lt 2 Views': ['X-Ray', 'Wrist'],
    'X-Ray Wrists BILATERAL 2 Views': ['X-Ray', 'Wrist'],
    'X-Ray Wrist Rt 2 Views': ['X-Ray', 'Wrist'],
    'X-Ray Wrist LEFT 2 Views': ['X-Ray', 'Wrist'],
    'X-Ray Wrists Bilat 3 Views +': ['X-Ray', 'Wrist'],
    'X-Ray Wrists Bilat 2 Views': ['X-Ray', 'Wrist'],
    'X-Ray Wrists BILATERAL 3 Views +': ['X-Ray', 'Wrist'],
    'X-Ray Scaphoid RIGHT': ['X-Ray', 'Wrist'],
    'X-Ray Scaphoid LEFT': ['X-Ray', 'Wrist'],
    'X-Ray Scaphoid Rt': ['X-Ray', 'Wrist'],
    'X-Ray Scaphoid Lt': ['X-Ray', 'Wrist'],
    'X-Ray Scaphoid L': ['X-Ray', 'Wrist'],

    # X-Ray Forearm Labels
    'X-Ray Forearm LEFT 2 Views': ['X-Ray', 'Forearm'],
    'X-Ray Forearm RIGHT 2 Views': ['X-Ray', 'Forearm'],
    'X-Ray Forearm Lt 2 Views': ['X-Ray', 'Forearm'],
    'X-Ray Forearm Rt 2 Views': ['X-Ray', 'Forearm'],
    'X-Ray Forearm Bilat 2 Views': ['X-Ray', 'Forearm'],
    'X-Ray Forearm BILATERAL 2 Views': ['X-Ray', 'Forearm'],

    # X-Ray Elbow Labels
    'X-Ray Elbow RIGHT 2 Views': ['X-Ray', 'Elbow'],
    'X-Ray Elbow LEFT 2 Views': ['X-Ray', 'Elbow'],
    'X-Ray Elbow Rt 2 Views': ['X-Ray', 'Elbow'],
    'X-Ray Elbow Lt 2 Views': ['X-Ray', 'Elbow'],
    'X-Ray Elbow RIGHT 3 Views +': ['X-Ray', 'Elbow'],
    'X-Ray Elbow Lt 3 Views +': ['X-Ray', 'Elbow'],
    'X-Ray Elbow LEFT 3 Views +': ['X-Ray', 'Elbow'],
    'X-Ray Elbow Rt 3 Views +': ['X-Ray', 'Elbow'],
    'X-Ray Elbows Bilat 2 Views': ['X-Ray', 'Elbow'],
    'X-Ray Elbows BILATERAL 2 Views': ['X-Ray', 'Elbow'],

    # X-Ray Arm Labels
    'X-Ray Humerus LEFT 2 Views': ['X-Ray', 'Arm'],
    'X-Ray Humerus RIGHT 2 Views': ['X-Ray', 'Arm'],
    'X-Ray Humerus BILATERAL 2 Views': ['X-Ray', 'Arm'],
    'X-Ray Humerus Lt 2 Views': ['X-Ray', 'Arm'],
    'X-Ray Humerus Rt 2 Views': ['X-Ray', 'Arm'],
    'X-Ray Humerus Bilat 2 Views': ['X-Ray', 'Arm'],

    # X-Ray Shoulder
    'X-Ray Shoulder RIGHT': ['X-Ray', 'Shoulder'],
    'X-Ray Shoulder LEFT': ['X-Ray', 'Shoulder'],
    'X-Ray Shoulder Lt': ['X-Ray', 'Shoulder'],
    'X-Ray Shoulder Rt': ['X-Ray', 'Shoulder'],
    'X-Ray Shoulder BILATERAL': ['X-Ray', 'Shoulder'],
    'X-Ray Shoulder Bilat': ['X-Ray', 'Shoulder'],
    'X-Ray Scapula RIGHT': ['X-Ray', 'Shoulder'],
    'X-Ray Scapula LEFT': ['X-Ray', 'Shoulder'],
    'X-Ray Scapula Lt': ['X-Ray', 'Shoulder'],
    'X-Ray AC Joints': ['X-Ray', 'Shoulder'],

    # X-Ray Abdo-Chest
    'X-Ray Chest 2 Views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest 1 View': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest 2 Views - obtain is cough': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest 4 Views +': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest 2 Views - consider only if cough or other respiratory problems': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest 2 Views - obtain is cough, chest pain, hypoxia, or clinical suspicion for pneumonia/ACS (see acute chest syndrome CPG)': [
        'X-Ray', 'Abdo/Chest'],
    'X-Ray Chest 3 Views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest/Abdomen 1view': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest/Abdomen 1 view': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest/Abdomen 2 views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Chest 2 vws/Abdomen 1 vw': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Clavicles Bilat': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Clavicle RIGHT': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Clavicles BILATERAL': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Clavicle LEFT': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Clavicle Lt': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Clavicle Rt': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Sternum 2 Views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Sternoclavicular Joints  Bilat 2 Views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Sternoclavicular Joints BILATERAL 2 Views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Ribs RIGHT': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Ribs LEFT': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Ribs Bilat': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Ribs BILATERAL': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Ribs Rt': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Ribs Lt': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Abdomen 3 Views +': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Abdomen 3 Views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Abdomen 2 Views': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Abdomen 1 View': ['X-Ray', 'Abdo/Chest'],
    'Xray Abdomen': ['X-Ray', 'Abdo/Chest'],
    'X-Ray Foreign Body Swallow Survey': ['X-Ray', 'Abdo/Chest'],

    # X-Ray Spine
    'Xray Cervical Spine (1 View Lateral)': ['X-Ray', 'Spine'],
    'Xray Cervical Spine (2-3 Views for C-Spine Clearance)': ['X-Ray', 'Spine'],
    'X-Ray Cervical Spine 2-3 Views': ['X-Ray', 'Spine'],
    'Xray Cervical Spine': ['X-Ray', 'Spine'],
    'X-Ray Lumbosacral Spine 2-3 Views': ['X-Ray', 'Spine'],
    'X-Ray Thoracic Spine 2 Views': ['X-Ray', 'Spine'],
    'X-Ray Thoracic Spine 3 Views +': ['X-Ray', 'Spine'],
    "X-Ray 3' Spine 2 Views": ['X-Ray', 'Spine'],
    'X-Ray Cervical Spine 1 view': ['X-Ray', 'Spine'],
    'X-Ray Cervical Spine 4-5 Views +': ['X-Ray', 'Spine'],
    'X-Ray Thoracolumbar Spine Spot 1 View': ['X-Ray', 'Spine'],
    "X-Ray Lumbosacral Spine 4-5 Views": ['X-Ray', 'Spine'],
    "X-Ray 3' Spine 1 View": ['X-Ray', 'Spine'],

    # X-Ray Head
    'X-Ray Skull & Sella': ['X-Ray', 'Head'],
    'X-Ray Skull Trauma': ['X-Ray', 'Head'],
    'X-Ray Skull Neurological': ['X-Ray', 'Head'],
    'X-Ray Facial Bones': ['X-Ray', 'Head'],
    'X-ray Nasal Bones': ['X-Ray', 'Head'],
    'X-Ray Mandible': ['X-Ray', 'Head'],
    'X-Ray Temporal Mandibular Joint RIGHT 2 Views': ['X-Ray', 'Head'],
    'X-Ray Orbits': ['X-Ray', 'Head'],
    'X-Ray Sinuses': ['X-Ray', 'Head'],
    'X-Ray Mastoids': ['X-Ray', 'Head'],
    'X-Ray Nasopharynx': ['X-Ray', 'Head'],
    'X-Ray Soft Tissue Neck': ['X-Ray', 'Head'],

    # X-Ray Bone Survey
    'X-Ray Rickets / Scurvy Survey': ['X-Ray', 'Bone Survey'],
    'X-Ray Skeletal Survey 12 Views +': ['X-Ray', 'Bone Survey'],
    'X-Ray Shunt Series 3 Views +': ['X-Ray', 'Bone Survey'],
    'X-Ray Bone Age Survey Hand': ['X-Ray', 'Bone Survey'],

    # X-Ray Other
    'X-Ray Image Intensifier Fluoro w Hard Copy': ['X-Ray', 'Other'],
    'X-Ray Image Intensifier Fluoro Setup': ['X-Ray', 'Other'],
    'X-Ray Consultation': ['X-Ray', 'Other'],
    'X-Ray Image Intensifier Fluoro Only': ['X-Ray', 'Other'],

    # CT--------------------------------------------------------------------

    # CT Head
    'CT Head Complex': ["CT", "Head"],
    'CT Orbits + Head': ["CT", "Head"],
    'CT Orbits + Head, w/o contrast': ["CT", "Head"],
    'CT  Head Complex w/o Contrast/CTV': ["CT", "Head"],
    'CT Head Complex, w/o contrast': ["CT", "Head"],
    'CT Facial Bones & Head Complex w/o contrast': ["CT", "Head"],
    'CT Head Complex /C-Spine w/o contrast': ["CT", "Head"],
    'CT Sinuses + Head': ["CT", "Head"],
    'CT Facial Bones': ["CT", "Head"],
    'CT Facial Bones, w/o contrast': ["CT", "Head"],
    'CT Orbits': ["CT", "Head"],
    'CT Orbits, w/o contrast': ["CT", "Head"],
    'CT Orbits, w & w/o contrast': ["CT", "Head"],
    'CT Sinuses': ["CT", "Head"],
    "CT Sinuses, w/o contrast": ["CT", "Head"],
    'CT Mandible w/o cont': ["CT", "Head"],
    'CT Petrous Bones': ["CT", "Head"],
    'CT Petrous Bones, w/o contrast': ["CT", "Head"],
    'CT Neck': ["CT", "Head"],
    'CT Neck, w/o contrast': ["CT", "Head"],
    'CT Head Complex w/o Contrast/CTA': ["CT", "Head"],

    # CT Head with Contrast
    'CT Facial Bones & Head Complex w contrast': ["CT", "Head with Contrast"],
    'CT Head Complex/Neck CTA': ["CT", "Head with Contrast"],
    'CT Head Complex/Neck w  contrast': ["CT", "Head with Contrast"],
    'CT Mandible w cont': ["CT", "Head with Contrast"],
    'CT Head Complex CTV': ["CT", "Head with Contrast"],
    'CT Carotid Arteries w contrast': ["CT", "Head with Contrast"],
    'CT Sinuses + Head, w contrast': ["CT", "Head with Contrast"],
    'CT Head Complex, w & w/o contrast': ["CT", "Head with Contrast"],
    'CT Head Complex, w contrast': ["CT", "Head with Contrast"],
    'CT Facial Bones, w contrast': ["CT", "Head with Contrast"],
    'CT Orbits, w contrast': ["CT", "Head with Contrast"],
    'CT Sinuses, w contrast': ["CT", "Head with Contrast"],
    'CT Sinuses + Head, w & w/o contrast': ["CT", "Head with Contrast"],
    'CT Sinuses, w & w/o contrast': ["CT", "Head with Contrast"],
    'CT Neck, w contrast': ["CT", "Head with Contrast"],
    'CT Orbits + Head, w & w/o contrast': ["CT", "Head with Contrast"],
    'CT Orbits + Head, w contrast': ["CT", "Head with Contrast"],
    'CT Petrous Bones, w contrast': ["CT", "Head with Contrast"],
    'CT Head Complex CTA': ["CT", "Head with Contrast"],

    # CT Trunk with Contrast
    'CT Neck/Chest/Abdo/Pelvis w contrast': ["CT", "Trunk with Contrast"],
    'CT Chest/Abdomen/Pelvis w contrast': ["CT", "Trunk with Contrast"],
    'CT Abdo,Pelvis w contrast': ["CT", "Trunk with Contrast"],
    'CT Abdo,Pelvis w/o contrast': ["CT", "Trunk"],

    # CT Chest
    'CT Chest': ["CT", "Chest"],
    'CT Chest, w/o contrast': ["CT", "Chest"],
    'CT Chest, w contrast': ["CT", "Chest with Contrast"],

    # CT Abdomen
    'CT Abdo': ["CT", "Abdomen"],

    # CT Spine
    'CT Spine Cervical w/o contrast': ["CT", "Spine"],
    'CT Spine Lumbar w/o contrast': ["CT", "Spine"],
    'CT Spine Thoracic w/o contrast': ["CT", "Spine"],
    'CT Spine Thoraco/Lumbar w/o Contrast': ["CT", "Spine"],

    # CT Pelvis
    'CT Pelvis': ["CT", "Pelvis"],
    'CT Pelvis, w/o contrast': ['CT', 'Pelvis'],

    # CT Femurs
    'CT Femurs (Bilat)': ["CT", "Femurs"],
    'CT Femurs (Bilat), w/o contrast': ["CT", "Femurs"],
    'CT Femurs (Bilat), w contrast': ["CT", "Femurs"],

    # CT Knees
    'CT Knees (Bilat)': ["CT", "Knees"],
    'CT Knees (Bilat), w/o  contrast': ["CT", "Knees"],

    # CT Leg
    'CT Tibia/Fibula': ["CT", "Tibia/Fibula"],
    'CT Tibia/Fibula, w/o contrast': ["CT", "Tibia/Fibula"],

    # CT Ankle
    'CT Ankle': ["CT", "Ankle"],
    'CT Ankle, w/o contrast': ["CT", "Ankle"],

    # CT Foot
    'CT Foot': ["CT", "Foot"],
    'CT Foot, w/o contrast': ["CT", "Foot"],

    # CT Shoulders
    'CT Shoulders (Bilat)': ["CT", "Shoulders"],
    'CT Shoulders (Bilat), w/o contrast': ["CT", "Shoulders"],
    'CT Clavicles (Bilat)': ["CT", "Shoulders"],
    'CT Clavicles (Bilat), w/o contrast': ["CT", "Shoulders"],

    # CT Elbow
    'CT Elbow': ["CT", "Elbow"],
    'CT Elbow, w/o contrast': ["CT", "Elbow"],

    # CT Wrist
    'CT Wrist': ["CT", "Wrist"],
    'CT Wrist, w/o contrast': ["CT", "Wrist"],

    # CT Hips
    'CT Hips (Bilat), w/o contrast':["CT", "Hips"],

    # CT Other
    'CT Neuro Minor Assessment': ["CT", "Other"],
    'CT Consultation Neuro': ["CT", "Other"],
    'CT Angio Extremities': ["CT", "Other"],
    'CT Consultation Body': ["CT", "Other"],
    'CT Body minor assessment': ["CT", "Other"],

    # ECG-------------------------------------------------------------------

    # ECG 12 Lead
    '12 Lead ECG': ['ECG', '12 Lead'],
    '12 Lead ECG Rhythm Strip': ['ECG', '12 Lead'],

    # ECG 15 Lead
    '15 Lead ECG': ['ECG', '15 Lead'],
    '15 Lead ECG Rhythm Strip': ['ECG', '15 Lead'],

    # ECH Holter
    '24 HR Holter': ["ECG", "Holter"],
    '48 HR Holter': ["ECG", "Holter"],

    # Ultrasound------------------------------------------------------------

    # Lower Extremity
    'US Lower Extremity (musculo-skeletal)': ["US", "Lower Extremity"],
    'US Lower Extremity Venous Assessment': ["US", "Lower Extremity"],
    'US Lower Extremity Arterial Assessment': ["US", "Lower Extremity"],

    # Upper Extremity
    'US Upper Extremity (musculo-skeletal)': ['US', 'Upper Extremity'],
    'US Upper Extremity Venous Assessment': ['US', 'Upper Extremity'],

    # Trunk
    'US Abdomen and Pelvis with Renal Doppler': ["US", "Trunk"],
    'US Abdomen and Pelvis with Rnl': ["US", "Trunk"],
    'US Abdomen and Pelvis with Rnl, Lvr Dop': ["US", "Trunk"],
    'US Abdomen and Pelvis': ["US", "Trunk"],
    'US Limited Abdomen and Pelvis': ["US", "Trunk"],
    'US Abdomen and Pelvis with Liver Doppler': ["US", "Trunk"],
    'US Limited Doppler Assessment Abdomen': ["US", "Trunk"],
    'US Transplant Liver': ["US", "Trunk"],
    'US Pelvis': ["US", "Trunk"],
    'US Hips': ["US", "Trunk"],
    'US Kidneys and Bladder': ["US", "Trunk"],
    'US Kidneys and Bladder with Doppler': ["US", "Trunk"],
    'US Transplant Kidney and Bladder': ["US", "Trunk"],
    'US Chest': ["US", "Trunk"],
    'AIUS Chest': ["US", "Trunk"],
    'US Breasts': ["US", "Trunk"],
    'US Diaphragms with M-Mode': ["US", "Trunk"],
    'US Spine': ["US", "Trunk"],

    # Neck
    'US Thyroid Gland with Doppler': ["US", "Neck"],
    'US Neck': ["US", "Neck"],
    'US Carotid Doppler Assessment': ["US", "Neck"],
    'US Neck, Chest Vascular with Doppler': ["US", "Neck"],

    # Head
    'US Face (lump)': ['US', 'Head'],
    'US Orbit': ['US', 'Head'],
    'US Brain with Doppler': ['US', 'Head'],
    'US Brain': ['US', 'Head'],

    # Transvaginal
    'US Transvaginal': ['US', 'Transvaginal'],

    # Testes
    'US Testes with Doppler': ['US', 'Testes'],

    # TTE
    '2D TTE structural': ['US', 'TTE'],
    'Non-cardiology 2D TTE structural': ['US', 'TTE'],
    '2D TTE functional': ['US', 'TTE'],
    'Non-cardiology 2D TTE functional': ['US', 'TTE'],
    'External Transthoracic Echo(TTE) - Other Centers': ['US', 'TTE'],

    # Other US
    'US Superficial Tissues': ['US', 'Other'],
    'US Consultation': ['US', 'Other'],
    'US Consultation Inpatient': ['US', 'Other'],
    'US No Show': ['US', 'Other'],

    # MRI-------------------------------------------------------------------

    # Head
    'MRI Brain w/o contrast': ['MRI', 'Head'],
    'MRI BRAIN W/O CONTRAST': ['MRI', 'Head'],
    'MRI Brain w & w/o contrast': ['MRI', 'Head'],
    'MRI Brain & Neck w/o contrast': ['MRI', 'Head'],
    'MRI Brain & Orbits w/o contrast': ['MRI', 'Head'],
    'MRI Brain & Orbits w & w/o contrast': ['MRI', 'Head'],
    'MRI Sella Turcica & Brain w & w/o contrast': ['MRI', 'Head'],
    'MRI Brain & Neck w & w/o contrast': ['MRI', 'Head'],
    'MRI Brain & IAC w & w/o contrast': ['MRI', 'Head'],
    'MRI Brain w contrast': ['MRI', 'Head'],
    'MRI Sella Turcica & Brain w/o contrast': ['MRI', 'Head'],
    'MRI Brain & IAC w/o contrast': ['MRI', 'Head'],
    'MRI Brain & Orbits w contrast': ['MRI', 'Head'],
    'MRI Face w/o contrast': ['MRI', 'Head'],
    'MRI Face w & w/o contrast': ['MRI', 'Head'],
    'MRI Orbits w & w/o contrast': ['MRI', 'Head'],
    'MRI Orbits w/o contrast': ['MRI', 'Head'],
    "MRI IAC'S w/o contrast": ['MRI', 'Head'],
    "MRI IAC'S w & w/o contrast": ['MRI', 'Head'],

    # Neck
    "MRI Neck w & w/o contrast": ["MRI", "Neck"],
    "MRI Neck w/o contrast": ["MRI", "Neck"],

    # Spine
    'MRI Cervical Spine w/o contrast': ['MRI', 'Spine'],
    'MRI Cervical Spine w & w/o contrast': ["MRI", "Spine"],
    'MRI Cerv Thoracic w/o contrast': ["MRI", "Spine"],
    'MRI C-T-L Spine w/o contrast': ["MRI", "Spine"],
    'MRI C-T-L Spine w contrast': ["MRI", "Spine"],
    'MRI C-T-L Spine w & w/o contrast': ["MRI", "Spine"],
    'MRI Lumbar Spine w/o contrast': ["MRI", "Spine"],

    # Chest
    'MRI Chest w & w/o contrast': ['MRI', 'Chest'],

    # Abdomen
    'MRI Abdomen w/o contrast': ['MRI', 'Abdomen'],
    'MRI Abdomen w & w/o contrast': ['MRI', 'Abdomen'],
    'MRI Liver w/o contrast': ['MRI', 'Abdomen'],

    # Upper Extremity
    'MRI Extremity Upper w/o contrast': ['MRI', 'Upper Extremity'],
    'MRI Extremity Upper W & w/o contrast': ['MRI', 'Upper Extremity'],

    # Lower Extremity
    'MRI Extremity Lower w/o contrast': ['MRI', 'Lower Extremity'],
    'MRI Extremity Lower W & w/o contrast': ['MRI', 'Lower Extremity'],
    'MRI Extremity Lower w & w/o contrast': ['MRI', 'Lower Extremity'],
    'MRI Extremity Lower  w/o contrast': ['MRI', 'Lower Extremity'],

    # Hips
    'MRI Hips w/o contrast': ['MRI', 'Hips'],
    'MRI Hips w & w/o contrast': ['MRI', 'Hips'],

    # Pelvis
    'MRI Pelvis w/o contrast': ['MRI', 'Pelvis'],
    'MRI Pelvis w & w/o contrast': ['MRI', 'Pelvis'],

    # Other
    'MRI Minor Assessment': ['MRI', 'Other'],
    'MRI Consultation Body': ['MRI', 'Other'],
    'MRI Consultation Neuro': ['MRI', 'Other'],
    'MRI Spectroscopy': ['MRI', 'Other'],

    # Tube Maintenance------------------------------------------------------
    'AI Nj Tube Maintenance - Check':['Tube Maintenance', 'GI'],
    'AI GJ Tube Maintenance - Change and Check': ['Tube Maintenance', 'GI'],
    'AI GJ Tube Maintenance - Check': ['Tube Maintenance', 'GI'],
    'AI G Tube Maintenance - Re-insertion': ['Tube Maintenance', 'GI'],
    'AI G Tube Maintenance - Check': ['Tube Maintenance', 'GI'],
    'AI GJ Tube Maintenance - Re-Insertion': ['Tube Maintenance', 'GI'],
    'GI GJ Tube Check': ['Tube Maintenance', 'GI'],
    'GI G Tube Check': ['Tube Maintenance', 'GI'],
    'AI G Tube Maintenance - Change and Check': ['Tube Maintenance', 'GI'],
    'GI NJ Tube Reinsert': ['Tube Maintenance', 'GI'],
    'GI NJ Tube Change & Check': ['Tube Maintenance', 'GI'],
    'GI NG Tube Insert': ['Tube Maintenance', 'GI'],
    'AI Nj Tube Insert': ['Tube Maintenance', 'GI'],
    'AI J Tube Maintenance - Change and Check': ['Tube Maintenance', 'GI'],
    'AI G Tube Maintenance': ['Tube Maintenance', 'GI'],
    'AI GJ Tube Maintenance': ['Tube Maintenance', 'GI'],
    'GI NJ Tube Insert': ['Tube Maintenance', 'GI'],
    'AI J Tube Maintenance - Re-Insertion': ['Tube Maintenance', 'GI'],

    # Picc Maintenance------------------------------------------------------
    'AI Picc Single Lumen Removal Int': ['Picc Maintenance'],
    'AI Picc Single Lumen Insertion Int': ['Picc Maintenance'],
    'AI Picc Single Lumen Removal & Insertion Int': ['Picc Maintenance'],
    'AI Picc Reposition': ['Picc Maintenance'],
    'AI Picc Double Lumen Removal & Insert Int': ['Picc Maintenance'],
    'AI PICC Line Maintenance': ['Picc Maintenance'],

    # Cecostomy Maintenance-------------------------------------------------
    'AI Cecostomy Maintenance - Change and Check': ['Cecostomy Maintenance'],
    'AI Cecostomy Maintenance - Re-Insertion': ['Cecostomy Maintenance'],
    'AI Cecostomy Maintenance - Check': ['Cecostomy Maintenance'],
    'AI Cecostomy Maintenance - Removal': ['Cecostomy Maintenance'],

    # GI Assessment---------------------------------------------------------

    # Colon
    'GI Colon Intussusception Reduction': ['GI Assessment', 'Colon'],
    'GI Colon Single Contrast': ['GI Assessment', 'Colon'],
    'GI Colon Double Contrast': ['GI Assessment', 'Colon'],

    # ESSB
    'GI Esophagus Stomach Duodenum Single Contrast': ['GI Assessment', 'ESSB'],
    'GI Esophagus Stomach Duodenum Double Contrast': ['GI Assessment', 'ESSB'],
    'GI ESSB Single Contrast': ['GI Assessment', 'ESSB'],
    'GI Esophagus Routine Including Pharynx': ['GI Assessment', 'ESSB'],
    'GI ESOPHAGUS STOMACH DUODENUM SINGLE CONTRAST': ['GI Assessment', 'ESSB'],

    # C-arm-----------------------------------------------------------------
    'c-arm': ['C-arm'],
    'C-arm': ['C-arm'],
    'C arm': ['C-arm'],

    # Uncategorized---------------------------------------------------------
    'AI CVL Repair': ['Other'],
    'AI Angio Cerebral': ['Other'],
    'AI Linogram Single Lumen Int': ['Other'],
    "NM GIS Meckel's": ['Other'],
    'Pacemaker Check (Single)': ['Other'],
    'RA FOREARM RT 2 VIEWS': ['Other'],
    'Unsolicited Remote (ILR/ICM)': ['Other'],
    'Event Recorder': ['Other'],
    'PET Consultation': ['Other'],
    'GU Minor Assessment': ['Other'],
    'GU Urethrogram Retrograde': ['Other'],
    'ILR/ICM Check': ['Other'],
    'AIUS Abdomen without Pelvis Limited': ['Other'],
    'Treadmill': ['Other'],
    'GI Loopogram Gastrointestinal': ['Other']
}

# categorizedData = DIseries.map(DI_tags)
# categorizedData.to_csv("result.csv")
