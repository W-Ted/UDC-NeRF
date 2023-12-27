from .udc_dataset_toydesk import ToydeskDataset
from .udc_dataset_scannet_multi import ScannetMultiDataset

dataset_dict = {
    'udc_dataset_toydesk': ToydeskDataset, # toydesk
    'udc_dataset_scannet_multi': ScannetMultiDataset, # scan multi
}
