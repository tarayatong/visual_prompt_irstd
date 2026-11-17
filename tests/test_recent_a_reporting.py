"""A completed baseline must match frozen test IDs and raw sufficient statistics."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scripts.report_recent_a_comparison import verified_summary
from visual_prompt.evaluation.comparison import image_metrics,summarize

class RecentReportingTests(unittest.TestCase):
    def fixture(self,folder):
        mask=np.zeros((8,8),bool);mask[3,3]=True
        rows=[image_metrics(mask,mask,'q1'),image_metrics(np.zeros_like(mask),mask,'q2')]
        score=summarize(rows)
        score.update(complete_test=True,epochs=120,manifest_sha256='frozen',actual_annotation_budget=50)
        path=folder/'summary.json';path.write_text(json.dumps(score))
        with (folder/'per_image.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return path,dict(manifest_sha256='frozen',test_ids=['q1','q2']),score

    def test_valid_counts_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            path,m,_=self.fixture(Path(d))
            self.assertEqual(verified_summary(path,m,120)['IoU_percent'],50.)

    def test_unexpected_or_missing_test_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,m,_=self.fixture(Path(d));m['test_ids']=['q1','other']
            with self.assertRaisesRegex(ValueError,'coverage'):verified_summary(path,m,120)

    def test_stale_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,m,_=self.fixture(Path(d))
            with self.assertRaisesRegex(ValueError,'protocol'):verified_summary(path,m,500)

    def test_summary_cannot_override_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path,m,score=self.fixture(Path(d));score['IoU_percent']=99.
            path.write_text(json.dumps(score))
            with self.assertRaisesRegex(ValueError,'Metric mismatch'):verified_summary(path,m,120)

if __name__=='__main__':unittest.main()
