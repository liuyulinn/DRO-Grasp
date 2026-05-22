pip install gdown 

gdown https://drive.google.com/uc?id=1XcBVply38NmM-B_QRHHR820N9NXxbAea
mkdir -p data/bodex

tar -xf DGN_xhand_right_lifted.tar.gz -C data/bodex
rm DGN_xhand_right_lifted.tar.gz