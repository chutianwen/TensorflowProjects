import re
import string
import timeit


s = "bromwell high is a cartoon comedy . it ran at the same time as some other programs about school life  such as  " \
    "teachers  . my   years in the teaching profession lead me to believe that bromwell high  s satire is much closer" \
    " to reality than is  teachers  . the scramble to survive financially  the insightful students who can see right" \
    " through their pathetic teachers  pomp  the pettiness of the whole situation  all remind me of the schools i " \
    "knew and their students . when i saw the episode in which a student repeatedly tried to burn down the school  " \
    "i immediately recalled . . . . . . . . . at . . . . . . . . . . high . a classic line inspector i  m here to" \
    " sack one of your teachers . student welcome to bromwell high . i expect that many adults of my age think that" \
    " bromwell high is far fetched . what a pity that it isn  t"

print(s)
exclude = set(string.punctuation)
translator = str.maketrans('', '', string.punctuation)
regex = re.compile('[%s]' % re.escape(string.punctuation))


def test_set(s):
    return ''.join(ch for ch in s if ch not in exclude)


def test_re(s):  # From Vinko's solution, with fix.
    return regex.sub('', s)


def test_trans(s):
    return s.translate(translator)


def test_repl(s):  # From S.Lott's solution
    for c in string.punctuation:
        s = s.replace(c, "")
    return s


print("sets      :", timeit.Timer('f(s)', 'from __main__ import s,test_set as f').timeit(1000000))
print("regex     :", timeit.Timer('f(s)', 'from __main__ import s,test_re as f').timeit(1000000))
print("translate :", timeit.Timer('f(s)', 'from __main__ import s,test_trans as f').timeit(1000000))
print("replace   :", timeit.Timer('f(s)', 'from __main__ import s,test_repl as f').timeit(1000000))
