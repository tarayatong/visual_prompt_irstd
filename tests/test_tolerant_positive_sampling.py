import numpy as np
from visual_prompt.core.candidate_proposals import point_instance_id
from visual_prompt.data.tolerant_positive_sampling import positive_regions, mask_supervision


def test_region_matches_scalar_tolerance_at_edges_ties_and_diagonals():
    labels=np.zeros((15,17),np.int32)
    labels[0,0]=1;labels[7,5]=2;labels[7,9]=3;labels[13:15,15:17]=4
    owner,distance=positive_regions(labels,3)
    for y,x in np.ndindex(labels.shape):
        assert int(owner[y,x])==point_instance_id(labels,y,x,3)
    assert owner[7,7]==2  # Equal distance:original row-major foreground order.
    assert owner[3,0]==1 and owner[3,3]==0  # Euclidean disk,not7x7 square.
    assert np.array_equal(owner[labels>0],labels[labels>0])
    assert np.array_equal(owner==0,distance<0)
    owner0,_=positive_regions(labels,0)
    assert np.array_equal(owner0,labels)


def test_offset_positive_uses_original_instance_and_ignores_other_targets():
    labels=np.zeros((20,20),np.int32);labels[8:10,8:10]=1;labels[8:10,15:17]=2
    owner,_=positive_regions(labels,3)
    assert labels[8,5]==0 and owner[8,5]==1
    positive=mask_supervision(labels,owner,8,5,20)
    assert positive['positive'] and positive['instance_id']==1
    assert positive['target'].sum()==4  # No dilation of the Mask label.
    # Crop x=-5..14;the second instance is outside. Move the positive crop so
    # both instances appear and verify the other one's pixels are ignored.
    both=mask_supervision(labels,owner,8,11,20)
    assert both['instance_id']==1 and both['target'].sum()==4
    assert (~both['valid']).sum()==4
    negative=mask_supervision(labels,owner,0,0,20)
    assert not negative['positive'] and negative['target'].sum()==0
    assert (~negative['valid']).sum()==4
