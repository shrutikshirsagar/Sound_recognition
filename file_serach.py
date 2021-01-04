import os, fnmatch
from shutil import copyfile



inp_path = '/media/amrgaballah/Backup_Plus/Internship_exp/try/a/'
src = '/media/amrgaballah/Backup_Plus/Internship_exp/Indoor_feat3/'
out = '/media/amrgaballah/Backup_Plus/Internship_exp/try/out/'
if not os.path.exists(out):
    os.makedirs(out)
for f in os.listdir(inp_path):
    print(f)
    filen = f.split('.')[0]
    print(filen)
    for root, dirs, files in os.walk(src):
        for basename in files:
            base = basename.split('_stat')[0]
            print(base)
            if fnmatch.fnmatch(base, filen):
                dst = os.path.join(root,basename)
                print(dst)
                final_filename = os.path.basename(dst)
                out_file = os.path.join(out, final_filename)
                copyfile(dst, out_file)
            else:
                continue
