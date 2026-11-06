"""分享包边界与Docker入口检查，不启动实际训练。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_supermimic', ROOT/'scripts/export_supermimic.py')
sharing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sharing)


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'source'
        self.root.mkdir()
        self.project = self.root/sharing.PROJECT
        for relative in ['scripts/export_supermimic.py', 'tests/test_supermimic_sharing.py']:
            p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# sharing tool fixture\n')
        for relative in ['README.md','tasks/mimic/state_train.py','logs/run/model.pt',
                         '.cache/warp/cache.bin','diagnostics/audit.json','docs/history/old.md',
                         'docs/architecture.md','data/reference/motion.npy',
                         'videos/sideflip_latest_cycle5800_20261106.mp4','videos/old.mp4',
                         'projects/accidental.txt','assets/accidental.txt']:
            p=self.project/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(relative)
        asset=self.root/'assets/unitree_model/Go2/mjcf'
        asset.mkdir(parents=True);(asset/'go2.xml').write_text('<mujoco/>');(asset/'LICENSE').write_text('original license')
        self.original_root=sharing.ROOT
        sharing.ROOT=self.root
        self.addCleanup(setattr,sharing,'ROOT',self.original_root)
        self.output=Path(self.tmp.name)/'export'

    def test_share_contains_required_data_but_excludes_local_experiment_state(self):
        output,manifest,_=sharing.export(self.output)
        paths={x['path'] for x in manifest['files']}
        self.assertIn('assets/unitree_model/Go2/mjcf/LICENSE',paths)
        self.assertIn('projects/go2_mujoco_mimic/tasks/mimic/state_train.py',paths)
        self.assertIn('projects/go2_mujoco_mimic/data/reference/motion.npy',paths)
        self.assertFalse(any(set(Path(p).parts)&{'logs','.cache','diagnostics','history'} for p in paths))
        self.assertFalse((output/sharing.PROJECT/'videos/old.mp4').exists())
        self.assertFalse((output/sharing.PROJECT/'projects').exists())
        for item in manifest['files']:
            self.assertEqual(item['sha256'],hashlib.sha256((output/item['path']).read_bytes()).hexdigest())
        self.assertTrue((self.project/'logs/run/model.pt').exists())

    def test_explicit_policy_export_verifies_hash_before_copying(self):
        d=self.project/'checkpoints';d.mkdir();p=d/'sideflip_policy_5800.pt';p.write_bytes(b'actor fixture')
        manifest={'policies':{'sideflip':{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}}
        (d/'demo_policies.json').write_text(json.dumps(manifest))
        sharing.export(self.output,with_demo_policies=True)
        self.assertEqual((self.output/sharing.PROJECT/'checkpoints'/p.name).read_bytes(),p.read_bytes())
        p.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            sharing.export(Path(self.tmp.name)/'invalid',with_demo_policies=True)

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir();(self.output/'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError):sharing.export(self.output)
        self.assertEqual((self.output/'keep.txt').read_text(),'keep')
        with self.assertRaises(ValueError):sharing.export(self.project/'export-inside-source')


class SuperMimicLauncherTests(unittest.TestCase):
    def test_root_alias_and_image_override_reach_the_state_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);capture=folder/'args.json'
            stub=folder/'docker'
            stub.write_text('#!/usr/bin/env python3\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            env={**os.environ,'PATH':f'{folder}:{os.environ["PATH"]}',
                 'CAPTURE':str(capture),'SUPERMIMIC_IMAGE':'custom:local','SUPERMIMIC_PYTHON':'/usr/local/bin/python'}
            env.pop('DISPLAY',None)
            subprocess.run(['bash',str(ROOT/'run.sh'),'supermimic','train-state','--task','sideflip','--cycles','2'],
                           cwd=folder,env=env,check=True,capture_output=True)
            args=json.loads(capture.read_text())
            self.assertIn('custom:local',args)
            self.assertEqual(args[args.index('--entrypoint')+1],'/usr/local/bin/python')
            self.assertIn('go2_mujoco_mimic.tasks.mimic.state_train',args)
            self.assertIn(f'{ROOT}/assets:/assets:ro',args)
            self.assertEqual(args[-4:],['--task','sideflip','--cycles','2'])


if __name__=='__main__':unittest.main()
