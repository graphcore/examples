# Graphcore

---

## Click Through Rate

Click through rate (CTR) is a binary classification problem. The inputs of this algorithm include User, Item and Context, and the output of this algorithm is whether the item will be clicked or the probability that the item will be clicked (Depends on whether the sigmoid function is used).        

This directory contains sample applications and code examples for typical CTR algorithms.


## The structure of this directory


| File                         | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| `din/`                       | DIN model definition, train and infer scripts, tests and other used code    |
| `dien/`                      | DIEN model definition, train and infer scripts, tests and other used code   |
| `common/`                    | The modules that can be used in several models                              |
| `requirements.txt/`          | Required packages                                                           |
| `README.md`                  | Structure of the folder and projects description                            |
| `LICENSE`                    | The license that applies to the files in the click_through_rate directory   |
| `NOTICE`                     | Notice of CTR projects                                                      |
