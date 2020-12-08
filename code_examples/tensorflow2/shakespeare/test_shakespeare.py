# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import unittest
import os
import pytest
import tensorflow as tf
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)

TEXT = """
I will tell you why; so shall my anticipation prevent your discovery,
and your secrecy to the King and Queen moult no feather. I have of
late, but wherefore I know not, lost all my mirth, forgone all custom
of exercises; and indeed, it goes so heavily with my disposition that
this goodly frame the earth, seems to me a sterile promontory; this
most excellent canopy the air, look you, this brave o'erhanging
firmament, this majestical roof fretted with golden fire, why, it
appears no other thing to me than a foul and pestilent congregation of
vapours. What a piece of work is man! How noble in reason? How infinite
in faculties, in form and moving, how express and admirable? In action
how like an angel? In apprehension, how like a god? The beauty of the
world, the paragon of animals. And yet, to me, what is this
quintessence of dust? Man delights not me; no, nor woman neither,
though by your smiling you seem to say so.

But soft, what light through yonder window breaks?
It is the east, and Juliet is the sun!
Arise fair sun and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
Be not her maid since she is envious;
Her vestal livery is but sick and green,
And none but fools do wear it; cast it off.
It is my lady, O it is my love!
O, that she knew she were!
She speaks, yet she says nothing. What of that?
Her eye discourses, I will answer it.
I am too bold, ?~@~Ytis not to me she speaks.
Two of the fairest stars in all the heaven,
Having some business, do entreat her eyes
To twinkle in their spheres till they return.
What if her eyes were there, they in her head?
The brightness of her cheek would shame those stars,
As daylight doth a lamp; her eyes in heaven
Would through the airy region stream so bright
That birds would sing and think it were not night.
See how she leans her cheek upon her hand.
O that I were a glove upon that hand,
That I might touch that cheek.
"""

with open(os.path.join(working_path, "shakespeare.txt"), "w") as f:
    f.write(TEXT)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TensorFlow2Mnist(SubProcessChecker):
    """Integration tests for TensorFlow 2 Shakespeare example"""

    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_default_commandline(self):
        self.run_command("python3 shakespeare.py",
                         working_path,
                         "Epoch 2/")
