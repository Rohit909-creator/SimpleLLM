from Data import Tokenizer, PrepData, MakeData
from Trainer import Trainer
from model import TLM
import torch
# print(dev,device)
text = """Wikipedia[note 3] is a free-content online encyclopedia written and maintained by a community of volunteers, collectively known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. Wikipedia is the largest and most-read reference work in history,[3][4] and has consistently been one of the 10 most popular websites.[5] Created by Jimmy Wales and Larry Sanger on January 15, 2001, it is hosted by the Wikimedia Foundation, an American non-profit organization.

Initially available only in English, versions in other languages were quickly developed. Wikipedia's combined editions comprise more than 61 million articles, attracting around 2 billion unique device visits per month and more than 15 million edits per month (about 5.8 edits per second on average) as of July 2023.[6][7]

Wikipedia has been praised for its enablement of the democratization of knowledge, extent of coverage, unique structure, and culture. It has been criticized for exhibiting systemic bias, particularly gender bias against women and geographical bias against the Global South.[8][9] While the reliability of Wikipedia was frequently criticized in the 2000s, it has improved over time, receiving greater praise in the late 2010s and early 2020s,[3][8][10][note 4] having become an important fact-checking site.[11][12] It has been censored by world governments, ranging from specific pages to the entire site.[13][14] Articles on breaking news are often accessed as a source of frequently updated information about those events.[15][16]"""

context_length = 100
batch_size = 32
n_embs = 512
mkdata = MakeData(text, context_length, batch_size)
print(mkdata.vocab_size)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)

m = TLM(mkdata.vocab_size, context_length, n_embs)
m.to(device)

trainer = Trainer(10, 2, device=dev)
trainer.train(mkdata,m)

idxx = torch.zeros((1,1), dtype = torch.long).to(device=device)

# print(decoder(m.generate(idxx, 1000)[0].tolist()))
print(mkdata.decode(m.generate(idxx, 1000)[0].tolist()))
