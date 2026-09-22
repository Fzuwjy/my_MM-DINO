import unittest
import torch
from research.cola.signal_audit import fusion_stats, phi_statistics, cosine


class SignalAuditTest(unittest.TestCase):
    def test_parallel_orthogonal_and_cancelling_fusion(self):
        a = torch.tensor([1.,0.])
        zero = torch.zeros_like(a)
        for b, expected in [(a,1.),(-a,0.),(torch.tensor([0.,1.]),2**-0.5)]:
            g = .5*a + .5*b
            result=fusion_stats(a,b,g,zero,zero,zero,[.5,.5])
            self.assertAlmostEqual(result['retention_vs_weighted_norm_sum'],expected)
            self.assertEqual(result['fusion_identity_max_error'],0)

    def test_phi_static_dynamic_and_distinct_files(self):
        a=torch.eye(2)
        static=phi_statistics([a,a,a],['a','a','b'])
        self.assertEqual(static['relative_centered_rms'],0)
        self.assertAlmostEqual(static['different_image_pair_cosine']['mean'],1)
        self.assertEqual(static['different_image_pair_cosine']['count'],2)
        dynamic=phi_statistics([a,-a],['a','b'])
        self.assertAlmostEqual(dynamic['relative_centered_rms'],1)
        self.assertAlmostEqual(dynamic['pair_cosine']['mean'],-1)
        self.assertIsNone(cosine(torch.zeros(2),torch.ones(2)))

    def test_capture_does_not_change_forward(self):
        from research.cola.model import CoLASegmenter, cpu_initialization
        from research.cola.signal_audit import Capture
        from dinov3.hub.backbones import dinov3_vits16
        torch.set_num_threads(1)
        with cpu_initialization(42):
            m=CoLASegmenter(dinov3_vits16(pretrained=False)).eval()
        x,s=torch.randn(1,3,64,64),torch.randn(1,1,64,64)
        with torch.inference_mode():
            expected=m(x,s)
            capture=Capture(m)
            on=capture.forward(x,s,True)
            off=capture.forward(x,s,False)
            capture.close()
            torch.testing.assert_close(expected,on['logits'],rtol=0,atol=0)
            torch.testing.assert_close(expected,off['logits'],rtol=0,atol=0)
            self.assertEqual(len(on['branches']),48)
            self.assertTrue(all(len(v)==2 for v in on['projected'].values()))


if __name__=='__main__':
    unittest.main()
