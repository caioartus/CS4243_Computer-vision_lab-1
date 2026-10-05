from gabor_branch import make_gabor_bank
from config import GaborConfig

config = GaborConfig()
res = make_gabor_bank(config=config)

print(res)
