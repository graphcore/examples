# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import pytest
import shutil
import subprocess
import sys

import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cwd, '..'))
import encode_dataset  # noqa: E402

ARTICLE = """ = = = 2000 – 2005 = = = 
 
 In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed " Scott Parry " in the episode , " In Safe Hands " . Boulter starred as " Scott " in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . A review of Boulter 's performance in The Independent on Sunday described him as " horribly menacing " in the role , and he received critical reviews in The Herald , and Evening Standard . He appeared in the television series Judge John Deed in 2002 as " Addem Armitage " in the episode " Political Expediency " , and had a role as a different character " Toby Steele " on The Bill . 
 He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . Boulter starred as " Darren " , in the 2005 theatre productions of the Philip Ridley play Mercury Fur . It was performed at the Drum Theatre in Plymouth , and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . Boulter received a favorable review in The Daily Telegraph : " The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn 's Hamlet ) , Robert Boulter , Shane Zaza and Fraser Ayres . " The Guardian noted , " Ben Whishaw and Robert Boulter offer tenderness amid the savagery . " 
 
 = = = 2006 – present = = = 
 
 In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which featured different playwrights , titled Burn / Chatroom / Citizenship . In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : " I loved working with a guy called Robert Boulter , who was in the triple bill of Burn , Chatroom and Citizenship at the National . He played my brother in Mercury Fur . " He portrayed " Jason Tyler " on the 2006 episode of the television series , Doctors , titled " Something I Ate " . Boulter starred as " William " in the 2007 production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a touching vulnerability to the stage as William . " 
 Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . Boulter portrayed a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ... who hits it off with Tammi " . Boulter guest starred on a two @-@ part episode arc " Wounds " in May 2008 of the television series Waking the Dead as character " Jimmy Dearden " . He appeared on the television series Survivors as " Neil " in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " . He portrayed an emergency physician applying for a medical fellowship . He commented on the inherent difficulties in portraying a physician on television : " Playing a doctor is a strange experience . Pretending you know what you 're talking about when you don 't is very bizarre but there are advisers on set who are fantastic at taking you through procedures and giving you the confidence to stand there and look like you know what you 're doing . " Boulter starred in the 2011 film Mercenaries directed by Paris Leonti . 
 """  # noqa: W291 W293

EXPECTED_LINES = [
    """ In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed " Scott Parry " in the episode , " In Safe Hands " . Boulter starred as " Scott " in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . A review of Boulter 's performance in The Independent on Sunday described him as " horribly menacing " in the role , and he received critical reviews in The Herald , and Evening Standard . He appeared in the television series Judge John Deed in 2002 as " Addem Armitage " in the episode " Political Expediency " , and had a role as a different character " Toby Steele " on The Bill .  He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . Boulter starred as " Darren " , in the 2005 theatre productions of the Philip Ridley play Mercury Fur . It was performed at the Drum Theatre in Plymouth , and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . Boulter received a favorable review in The Daily Telegraph : " The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn 's Hamlet ) , Robert Boulter , Shane Zaza and Fraser Ayres . " The Guardian noted , " Ben Whishaw and Robert Boulter offer tenderness amid the savagery . " """,
    """ In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which featured different playwrights , titled Burn / Chatroom / Citizenship . In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : " I loved working with a guy called Robert Boulter , who was in the triple bill of Burn , Chatroom and Citizenship at the National . He played my brother in Mercury Fur . " He portrayed " Jason Tyler " on the 2006 episode of the television series , Doctors , titled " Something I Ate " . Boulter starred as " William " in the 2007 production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a touching vulnerability to the stage as William . "  Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . Boulter portrayed a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ... who hits it off with Tammi " . Boulter guest starred on a two @-@ part episode arc " Wounds " in May 2008 of the television series Waking the Dead as character " Jimmy Dearden " . He appeared on the television series Survivors as " Neil " in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " . He portrayed an emergency physician applying for a medical fellowship . He commented on the inherent difficulties in portraying a physician on television : " Playing a doctor is a strange experience . Pretending you know what you 're talking about when you don 't is very bizarre but there are advisers on set who are fantastic at taking you through procedures and giving you the confidence to stand there and look like you know what you 're doing . " Boulter starred in the 2011 film Mercenaries directed by Paris Leonti . """
]

EXPECTED_PATHS = [
    f"datasets/wikitext-fake-gpt2/seqlen_128/seqlen_128.{filename}.npy"
    for filename in (
        f"test.cache.np_{np.__version__}",
        f"train.cache.np_{np.__version__}",
        f"valid.cache.np_{np.__version__}",
        "vocab")]


@pytest.mark.category1
def test_load_article():
    mock_file_iterator = ARTICLE.split("\n")
    sequences = encode_dataset.load_articles(mock_file_iterator)
    assert len(sequences) == 2

    for i, seq in enumerate(sequences):
        assert seq == EXPECTED_LINES[i]


@pytest.mark.category1
@pytest.mark.requires_encoder
def test_raw_encoding(gpt2_repo_path):
    encoder = encode_dataset.get_encoder(gpt2_repo_path)

    for i, seq in enumerate(EXPECTED_LINES):
        decoded = encoder.decode(encoder.encode(seq))
        assert decoded == seq


@pytest.mark.category1
@pytest.mark.requires_encoder
@pytest.mark.parametrize(
    "seq_len, surviving_sequences, num_processes",
    [
        (5, (0, 1), 1),
        (128, (0, 1), 1),
        (256, (0, 1), 1),
        (400, (1, ), 1),
        (128, (0, 1), 2),
        (400, (1, ), 2)
    ]
)
def test_cropped_encoding(gpt2_repo_path, seq_len, surviving_sequences, num_processes):
    encoder = encode_dataset.get_encoder(gpt2_repo_path)
    tokens = encode_dataset.encode(encoder, EXPECTED_LINES, seq_len, num_processes)

    assert len(tokens) == len(surviving_sequences)

    for i, t in enumerate(tokens):
        expected_line = EXPECTED_LINES[surviving_sequences[i]]
        assert len(t) == seq_len
        decoded = encoder.decode(t)
        assert expected_line.startswith(decoded)


@pytest.mark.category1
@pytest.mark.requires_encoder
def test_output_location(gpt2_repo_path):
    np.random.seed(0xdeadcafe)
    fake_dir = "datasets/wikitext-fake-raw"

    def gen_fake_dataset(vocab_size, seq_len):
        alphabet = [chr(c) for c in range(ord('a'), ord('z')+1)]
        words = np.random.choice(alphabet, size=(vocab_size, 5))
        vocabulary = [''.join(word) for word in words]

        def gen_fake_article():
            article = [f"= = = {np.random.choice(vocabulary)} = = =", ""]
            article.append(
                ' '.join(np.random.choice(vocabulary, size=(2*seq_len))))

            return '\n'.join(article)

        os.makedirs(fake_dir, exist_ok=True)
        for filetype in ("train", "test", "valid"):
            with open(f"{fake_dir}/wiki.{filetype}.raw", "w") as outfile:
                for _ in range(2):
                    print(gen_fake_article(), file=outfile)
                    print(file=outfile)

    gen_fake_dataset(1000, 128)

    cmd = [
        sys.executable,
        "encode_dataset.py",
        "--sequence-length", "128",
        "--gpt2-repo-path", gpt2_repo_path,
        "--dataset-name", "wikitext-fake"]
    env = os.environ.copy()
    env["PYTHONPATH"] += ":./"
    result = subprocess.run(cmd, env=env)
    assert result.returncode == 0

    paths = [os.path.exists(path) for path in EXPECTED_PATHS]
    assert all(paths)

    # Clean the fake raw dir and generated dir with numpy files
    shutil.rmtree(fake_dir)
    shutil.rmtree(fake_dir.replace("raw", "gpt2"))
