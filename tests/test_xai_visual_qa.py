"""Fictional QA only: no real paths, trained state, clinical rows or patient inputs."""
import copy
import json
import unittest
import numpy as np
from ele630.xai_visual_qa import (audit_maps, render_preview, public_summary,
    validate_public, midpoint, anatomy_slice, map_stats)

class VisualQATests(unittest.TestCase):
    def fixture(self, g=1., h=1.):
        z,y,x=np.indices((9,33,35));image=((x+y+z)%90/90).astype(np.float32)
        maps={m:{"signed":np.full((3,3,3),v),"positive":np.full((3,3,3),max(v,0))} for m,v in (("gradcam",g),("hirescam",h))}
        a=np.array([[0,-.8,0,9],[1.2,0,0,-7],[0,0,3,4],[0,0,0,1.]])
        return image,maps,a
    def test_zero_maps_preserved(self):
        image,maps,a=self.fixture(0,0);private,public,cuts=audit_maps(image,maps,a,modality="ncct")
        self.assertFalse(public["gradcam_nonzero_3d"]);self.assertFalse(public["gradcam_positive_clipping_empty"])
        self.assertFalse(public["suitable_for_report"])
        self.assertTrue(render_preview(image,cuts,a,modality="ncct").startswith(b"\x89PNG"))
    def test_tiny_gradcam_shared_scale_suppression(self):
        image,maps,a=self.fixture(1e-9,1);private,public,cuts=audit_maps(image,maps,a,modality="ctp")
        self.assertTrue(public["gradcam_nonzero_3d"]);self.assertTrue(public["gradcam_nonzero_slice"])
        self.assertTrue(public["gradcam_shared_scale_suppressed"])
        before=copy.deepcopy(cuts)
        for separate in (False,True):
            self.assertTrue(render_preview(image,cuts,a,modality="ctp",separate=separate).startswith(b"\x89PNG"))
        for k in cuts:np.testing.assert_array_equal(cuts[k],before[k])
    def test_negative_clipping_is_not_true_zero(self):
        image,maps,a=self.fixture(-1,1);private,public,_=audit_maps(image,maps,a,modality="ncct")
        self.assertTrue(public["gradcam_positive_clipping_empty"])
        self.assertGreater(private["methods"]["gradcam"]["signed"]["feature"]["nonzero"],0)
        self.assertFalse(public["gradcam_nonzero_3d"])
        self.assertTrue(public["gradcam_signed_nonzero_3d"]);self.assertTrue(public["gradcam_signed_nonzero_slice"])
    def test_volume_nonzero_slice_zero(self):
        image,maps,a=self.fixture(0,1)
        maps["gradcam"]["signed"][2]=1;maps["gradcam"]["positive"][2]=1
        _,public,_=audit_maps(image,maps,a,modality="ctp")
        self.assertTrue(public["gradcam_nonzero_3d"]);self.assertFalse(public["gradcam_nonzero_slice"])
    def test_affine_alignment_and_no_clinical_orientation_claim(self):
        image,maps,a=self.fixture();private,public,cuts=audit_maps(image,maps,a,modality="ncct")
        np.testing.assert_array_equal(private["affine"],a)
        self.assertAlmostEqual(private["pixel_aspect"],.8/1.2)
        self.assertTrue(public["image_cam_array_alignment_verified"])
        self.assertFalse(public["physical_display_orientation_verified"])
        with self.assertRaises(ValueError):audit_maps(image,maps,np.zeros((4,4)),modality="ncct")
        maps["gradcam"]["positive"]=np.zeros((2,2,2))
        with self.assertRaises(ValueError):audit_maps(image,maps,a,modality="ncct")
    def test_asymmetric_projection_uses_same_native_indices_without_affine_flip(self):
        image,maps,a=self.fixture();before=image.copy()
        feature=np.arange(27,dtype=float).reshape(3,3,3)
        for data in maps.values():data["signed"]=feature.copy();data["positive"]=feature.copy()
        a[:3,0]*=-1
        private,public,cuts=audit_maps(image,maps,a,modality="ncct")
        self.assertEqual(cuts["gradcam"][0,0],feature[1,0,0])
        self.assertEqual(cuts["gradcam"][16,16],feature[1,1,1])
        self.assertEqual(cuts["gradcam"][32,34],feature[1,2,2])
        self.assertTrue(public["hirescam_nondegenerate"])
        np.testing.assert_array_equal(image,before)

    def test_frame_positive_and_aggregate_positive_are_distinct(self):
        image,maps,a=self.fixture(0,0)
        for data in maps.values():data["positive"][:]=1
        with self.assertRaises(ValueError):audit_maps(image,maps,a,modality="ctp")
        private,_,_=audit_maps(image,maps,a,modality="ctp",aggregate=True)
        self.assertEqual(private["positive_definition"],"mean of per-frame clipped maps")
    def test_public_schema_rejects_private_numeric_and_arbitrary_content(self):
        safe=public_summary("ctp");self.assertEqual(validate_public(safe),safe)
        for key,value in (("patient_id","fictional"),("gradcam_nonzero_3d",.1),("uncertainty",["private/path"]),("suitable_for_report",True)):
            bad=copy.deepcopy(safe);bad[key]=value
            with self.assertRaises(ValueError):validate_public(bad)
        self.assertNotIn("ordinal",json.dumps(safe));self.assertNotIn("logit",json.dumps(safe))
    def test_selection_deterministic_and_requires_independent_support(self):
        self.assertEqual(midpoint((9,33,35)),4)
        mask=np.zeros((9,33,35),dtype=bool);mask[3:6,5:20,5:20]=True
        with self.assertRaises(ValueError):anatomy_slice(mask)
        self.assertEqual(anatomy_slice(mask,independently_verified=True),4)
    def test_nonfinite_count_is_private_and_invalid_projection_refused(self):
        self.assertFalse(map_stats([np.nan,0,1])["finite_valid"])
        image,maps,a=self.fixture();maps["gradcam"]["signed"][0,0,0]=np.nan
        with self.assertRaises(ValueError):audit_maps(image,maps,a,modality="ncct")

    def test_original_manifest_hash_source_and_schema_are_bound(self):
        import hashlib
        from ele630.xai_visual_qa import verify_original_manifest
        data=b"fictional PNG"; protocol="a"*64
        prior=dict(schema=1,status="PRIVATE_REVIEW_ONLY",modality="ncct",
            source_revision="79dd321f0deff7e6028da9081699c64c56e56d63",protocol_sha256=protocol,
            validation_count=23,selected_development_ordinal=11,
            selection="median absolute frozen fused logit; original ordinal tie-break",
            plane="native Z midpoint (unoriented array view)",ctp_original_frame_index=None,
            cam_target="original predicted-class fused support logit",cam_layer="stage4.conv2",
            exported_to_external_service=False,filename="ncct_development_example.png",
            png_sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(verify_original_manifest(prior,data,modality="ncct",protocol=protocol),prior)
        for key,value in (("source_revision","bad"),("png_sha256","bad"),("selected_development_ordinal",True),("fusion_state_artifact","unknown")):
            bad=dict(prior);bad[key]=value
            with self.assertRaises(ValueError):verify_original_manifest(bad,data,modality="ncct",protocol=protocol)
    def test_private_write_is_exclusive_and_names_are_fixed(self):
        import importlib.util,tempfile
        from pathlib import Path
        script=Path(__file__).resolve().parents[1]/"scripts/experiments/export_stage7_private_overlay.py"
        spec=importlib.util.spec_from_file_location("qa_export_test",script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            module.exclusive_write(folder,"private_qa.json",b"fictional")
            with self.assertRaises(FileExistsError):module.exclusive_write(folder,"private_qa.json",b"replacement")
            with self.assertRaises(ValueError):module.exclusive_write(folder,"../secret.json",b"fictional")

if __name__=="__main__":unittest.main()
