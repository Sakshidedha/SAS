
import streamlit as st #importing the streamlit
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px

# Set the title and icon of the Streamlit web page
st.set_page_config(page_title="Sentiment Analysis System", page_icon="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAPDw8PDw8PDw8PDw0PDw8PDw8PEA8QFRUWFxURFRUYHSggGBolHRUVITEiJSkrLi4uFyAzODMtNygtLisBCgoKDg0OGxAQGy0lICUtLS0wLS0tLS0tLy0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAKwBJAMBEQACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAAAQUEBgcDAgj/xABDEAABAwICBgUJBgMJAQEAAAABAAIDBBEFIQYSMUFRYRMiMnGRBxZCUlOBscHRFCNicpKhM0PhFSSCorLC0vDxc2P/xAAaAQEAAgMBAAAAAAAAAAAAAAAABAUBAgMG/8QANREAAgIBAgQEBQMDAwUAAAAAAAECAwQREgUhMUETFSJRFDJSYYFxkbEjQqEGM8EkNERT8P/aAAwDAQACEQMRAD8A7ggCAIAgCAIAgCAIAgCAIAgCAICLrAF1hyS6g+TIOI8QtXbBdWv3M7WR0g9ZviE8WHuv3G1+x9a3ctlOL7oxowsglZAQBAEAQBAEAQBAEAQBAEAQBAEBKAIAgCAIAgCAIAgCAIAgPCoq44+29re85+Cj25VVS1nJI3hXOfyoq59Iox2Gufz7I+qqLv8AUFMeUE3/AATIcPsfzcjAm0gmd2Q1nu1iqy3j2RL5UkSY8Pgur1MOTEpnbZXe46vwUCziWVPrNkiOLUuxjulcdrnHvJKjSvtl1k/3OiriuiPhc98vdm21BN0vcaIkOI2Ej3lZVs10b/cw4RfY9o6yVuyR4/xFd45t8Ok2aOit9Yoyosbnb6Qd+YAqbXxrKh1ev6nGWFU+i0M6DSQ/zI/ew/Iqwp/1F/7I/sRp8O+lllTYvDJsfqng/qn6K3o4rjXdJaP78iJPFth1RnAqxUk1qiOSsgIAgCAIAgCAIAgCAIAgJQBAQgCAIAgCAIAUAQFdXYvFFlfXd6rc/E7lV5fFqMflrq/ZEmrFss6FHV43LJk09G3g3b4rzmTxq+3lH0r/ACWNeFXHrzZWuJOZNzxKqJzcnrJ6kxRSWiIWpkICUB9QwuedVjS48AutVFlstsFqzWc4wWrZZswRwF5ZI4+RNyrePBZJa2zUSG85PlCLZP8AY7DkypjJ4G31W3lFUvktWpj4ya6wZiVmGSxZubdvrNzCg5PDL6Fq1qvdHarKrs5LqYagEkLACAIDIpq2WLsPIHDa3wUzHz76H6Jfg42Y8J9UXVHpCDYSt1fxNzHvG5egxePxlyuWn3RX24DXOBdwzNeNZrg4HeDdegqthZHdB6ogSi4vRo+10NQgCAlAQUAQBAEAQBASgIQBAEAQBAEAQFfi+MQ0rdaV1ieywZvd3BcrboVrWTJGPi2Xy0gjUfOqSqc5luib6LQes4fiP0XmOKZt0o+h6RLtcMjTFSfNnyvON6nQlYBCAlAEB6UtO6V7WN2u/YbypGNjzvsVce5ztsVcXJlrV1jaYdDBbX/mSHM3V1k5cMKPgY/zd2QaqZXvfZ07Ip5Hlxu4lxO8m6obLZWPWT1LCMVFaJHyQtEzYzqHE5Ija+uzex2eXLgrLE4nbQ9G90fZkW7FhYtVyZ7YnRsLRPD/AA3dpvqFSOIYcHWsmj5X1Xsc8e6Sl4VnUq1SE4IAsgLACyD6Fe6nBka4tsL23O5WUrFyLaprw3oRcx1xqcrOxaYFpjHNZk4EMhyDr/du9/onvXsMXicbPTZyf+DzFWVGT0fI2kFWiepKJWQEAQBAEAQBAEAQBAEAQBAEAQGq6S6XNgvFBaSbMOdtZH9TyUDJzVDlHmy3weFyu0nZyj/Jz6pqHyvMkji97trnG5VRKbk9WelrqjXHbBaI+Y5C0hzdrSCFznHfFxfc3lFSWhtFNOJGhw37uB3heeurdctrKqcHB6M9VyNCUBCAIC4wo9FBNP6XYZyOXzP7K/4dpRi2ZHfoivyf6l0azEwqWNsutMLtscyNYa3EjxULh1tMb91/T/k75MJyr0rPPEHxulcYhZmVsrDnkuWfOqdzlV0N8eM4wSn1MZQjsWrp6f7LqBo6Ww3dbW463BXsr8P4LYl6/wDkgKu7x9deROASaxfA7sytdl+ID6fBODW7nLHl0kmM2OiVi7FXIwtcWna0kH3Kmthsm4vsTYS3RTPlczYIAgCAo8ZqtZ3Rjst283f0U+ivatWeV4xmb5+FHoitUjUpDYdHtKJaUhkl5YPVJ6zPynhyVjicQnT6Zc0Sachw5PodGoa2OdgkicHsO8fAjcV6Oq2Nkd0XyLOM1Jao911NiUAQBAQgJQBAEAQEIAgCAEoDQ9K9LNbWgpnWbmJJQczxaw8OaqsrM/th+56Hh/DNNLLfwjS1WF/0JQEICxwWq1Hah7L9nJ25Qc2jfHcuqI2TXuW5F+qUrwgCAIC320GW6XPx/wDFfdeFcvq/5IH/AJf4KhURPCALBkJqYM/Ah/eYrcXf6SrTg+vxcSLm6eEzwxA/fS//AEf8VGzn/wBRPT3OtH+3H9DHUQ6hAEBjYhU9Gwn0jk3v4rtTDdIg8Qylj1N93yRrZKsDxLbb1ZCGCVkFhguMS0kmvGbtNteM9l4+R5qTjZU6Jax6ex1rtlW9UdPwfFY6qMSRnk5p7THcCvU4+RC6O6Ja12Ka1RnrudAgCAhASgIQEoAgIQBACsA0HTHSbX1qand1BdssjT2+LAeHFVWZla+iDPQ8M4dppbYv0RpqrS/CAIAgCA2XDKrpGC/abk76qhyqfDn9isur2SMxRTiQgCAt8FcJGS07jbXGsz8w/wDAVfcKlG2qeNLvzRAy04TjauxXMaGSASg2a6zxv5qrhBVXKNy5J8yU5OcNYdS46eg9T/I9Xvj8J+n/AAyBsy/cdPQeof0vTx+E/T/hjw8v3MLE5KYtb0DSHXzyIFveq/PswpQXw60f6EjHjepf1HyPfBmCJklS7Y1pay+8n/tlI4XBUVTyp+3I55UvEkqolQSSSTtJJPeVRTluk2+5PitFoQtTIQBZ0MNpLVmuYjU9I/Lstyb9VY1Q2RPF8RynkW6rouhiLoV4QBAEBm4RiclLKJIzyc09l7eBUjHyJUz3ROldjg9UdUwjE46qJssZyOTmntMdvaV6vHvjdDdEt65qa1RmrubhAEAQBASgCAICEBp+m+kPRA00LvvHD71w2safRHMquzMnatkepdcLwPEfizXLt9zn6qD0wQBASgCAhAZWH1XRPB9E5O7uKj5NPiQ07nK6G+JswzVA1oVZKwCLID7ikLSHNNiDcHmulVsq5qcXo0ayipLRlw4R1gBuI6gDMHY9egkqOJQ11UbF/krk54z6axK6ow+aM2dG7vaC4eIVRdw/Ipekov8ABMhk1S6M8WQPOQY8nk0rjHHtk9FF/sdHbBdWWVNhBA16giNg3E9Y8uStcfhO1eJkval29yHZl6+mrmzwxOv6WzGDViZk1vHmVw4hneNpXWtILodcbH2eqXVmAqolBAEBXYxVajdQdp+3k3epOPXq9zKXi+Z4dfhxfNlEpp5QIAgCAIAgLTR/GH0coeLmN1hKz1m8e8KXh5UqJ69u52ptdctTqtLUMlY2SM6zHgFpG8L1kJxnFSj0ZbxkmtUey3MhAEAQEIAgCAqNJcYFJAX5GR3Vibxdx7go+RcqoakzCxXkW7e3c5RLI57nPcS5ziXOJ2knaVQSbk9X1PZQioRUUfKwbBAEBKAIAgIQF7glVrN6M7W7Obf6Knzqdst66MgZNej3Is1XkUIAgJBWVJp6oNJmbBi07Mg+44OGsrGri2VWtFLX9SNPEql2PV2Ozne0cw1dpcbymtFovwaLBqMGed8hu9xceZVddk23PWcmyTCuMPlWh5LgbkoAgPiWQNaXHYBcraMXJ6I522KuDlLsaxUzGRxcd58BuCs4x2rQ8Lk3u+xzZ5LJwCAIAsgICUBCwDa9CMc6KT7PIfu5D1CfQkO7uPxVvwzL2S8OT5MmYt2j2s6GF6MsiUAQBAEAQHw9wAJJsALk8AsN6LmEm3yOT6TYsauoc8fw29SIfhHpd52qgybvEnr2PZcPxfh6ku75sq44y4hrQXOcQA0C5J4BcEnJ6ImSkordLobNBo3DAwSYhP0d8xCw3eeRP08VOWLCta3P8FRPiFtstuNHX7knFMLZkyifIPWe7M+JTxcZdI6mPhc6fOVmgbLhNR1THLSuOxwN2j4j9lndjWcmtBsz6eaakYONaOSU7elY4T05zErNw/EPmuN2K4LcuaJOLxGFr2TW2XsUqiliEAQH3BMWODxtB8eS0sgpxcWazhuWhtMMoe0OGwi/9F52yDhLayqknF6M+1oahAS0X2Z8hmt4wlLojDkl1PZtJKdkb/0ld44d8ukGaO6tf3Ik0Mo/lSfpKy8DJ+hmvxFf1I8nxOb2muHeCFylRZH5otfg3VkX0Z8LibhAEBTY1VXPRjYM3d+4Kbj16LczzXGcvc/Bj+SqUkoCUMEIDKw7D5aiQRxN1nHM7g0cSdwXemid0tsTeFcpvRGwuwmgo8quZ082+KLIDkbfMhWPw2LR/uy1fsiT4dVfzvVnx/a+G7BQG3EkX+K1+KxO1Y8Wlf2n2yiw2r6sMj6WU9lsmbSeGZ+BWVViX8oPaxsps+V6Mo8XwiWlfqyjI9h4za/uPyULIxZ0S0l+5wsqlB8zBUZcjmdQ0Qxf7VAA4/exWZJz4P8Af8QV6rh+T41fPqi2x7d8S9U87koAgCAgoDVdPcV6KEQNPXnve20Rjb47PFQM67ZDaurLbhGL4tu99I/yc5VMeqNtwhjMPpPtsjQ6om6tO0+i0+l8zysrGpKivxH1fQo8mUsu/wACPyrqUVRT1VQ19W9skjbnWlIyFuHADkoso2TXiNalhXZj0yVMXo/Yr1wJh9wQukc1jGlznGzWgXJKzGLk9EaznGEd0nyRfYTiM2Hy9DUMcIX26SJ+YDXem3/ual1WSoltn0KzIoqy4eJU/Uu54aUYSKaYGPOCYdJEdoA3tvyv4ELXKp8OWq6M68PyndDSXzLkymUUsAgCAtMEqrHozsdm3k7gq7Oo3LfHqRMqvluNtosFllsSOjbxdtPcFnF4NffzlyRS25tcOS5suqbAoWdoGQ/iOXgF6CjgmNXzktz+5X2ZtkunIsYoWtFmta3uACs4U1w+VJEZyk+rPRdTUICCL81q4p9UNTDqMNhk7UYvxHVP7KHdw7Ht+aKO0MiyPRlTV6OkZxOv+F/1VJk/6f70y/DJtfEPrRr2KF1O067S12xoO88uKo5Yllc9k1odsnNhVS7E/wBDVXOJJJzJNyeal6aHipTc5OT6shDUID7ijL3NY0Xc4hrRxJyAW8IuT0Xcyk29EbZitUMNgbSU5/vEjQ6eUbRfcOfDgO9W91ixK1VX8z6smWSVMdkeprdZh88Qa+aN7BJmHOHaO3Pn3qstpsgt011Is65R5tGIuBoZNFQSzkthjdIWi51RsC7VUzsekFqbwhKXQ2PAK8VDXYdWXIddsL3duN49G538PBWWLd4qePd+P1JVU9/9OZrdfSOglfE/tMcWnmNxHeLFVl1Tqm4PsRZxcJOLM7RrFPstSx5P3bupKPwnf7tqkYOQ6bU+3c3x7Nkzq7V6xPkXBKyCUAQHy42WGNDkekWIfaamSS/VvqR/kbs+Z968/k2eJY2e0wKFTSo9+5g0sPSSMZ672N8SAuUI6ySJFstsHL2RsGnk96lsLcmQRMa0bgSLn9tVS8+XrUV2RW8Hh/Sdj6tmNT6SyspHUgawtLXMD89YMde4tv2nNaRypRr8NI6WcNhO/wAbX8FGopZGVhla6nmZMwAuYSbHYQRYjwK6VWOuSkjjkUK+twl3MjHcXfWSiR7Ws1WhrWtzsMzt962vvd0tWcsPEjjQcU9S3nPTYOxzs3U02oD+G9gPBw8FKl68VN9iBBeFxBpdJI1dV5dEIDKw+gkqJBHE0ucdvBo4k7guldcrHpE433wohumzo2j+isNMA94Es3rkZN/KN3ftVxRhxr5vmzy2ZxGy96LlH2NgAUwriUBCAlAEBCAlAQgMavoY52FkrA9p3HaDxB2grlbTCxaSWprKKktGc90j0WfTXkivJBv3vj7+I5rz2Xw+VXqhzRW3Yzhzj0NcVWRQgL/QenElawnZGyST3iwH+pWHDIKV6b7cyTix1s/QrMTq3S1EspObpHOHIA2b+wCj32uVrl9zlOWs2zPxvSSWrjZG9jGhhDiW3Os61r8tq75OdO+Ki0dLb3YtGUqgEctcAx19EXljGvEgF2uuMxexuO8qZiZksfXRa6nem519DBmrHumM5NpDJ0lxlZ175Li7W7PE766nPe3LcX+nTA6SnqALdPA0nvGd/Bw8FP4nFOUbPdEjKXNS9zWFVoiHT9C8R6ela1xu+H7t3Egdk+HwXqeG3+LSteqLbGs3QL9WBIJQBAUemNf0FJIQbPktEzvdtPhdRcuzZWydw6jxb4rsuZypUJ7I96CUMmiedjJI3HuDgSt63pNM5Xx3VSX2ZdaeQlta526RkbweOWr8lJzk/F19yv4PLXH07plDFG57g1jS5zjYNaCSTyCiJOT0RZynGC1k9EX9NoXWPFyI477nvz8ACpkcG18ysnxjHi9FqzDxPRyqpgXPjuwbXsOs0d+8LnZi2Q6o70cRoueiej+5UqMTjaWfd4K6+2eoGrzAcP8AgVP024v6spW/E4jy7I1dQC6MrDMOkqZWxRi7jmTua3e48l0qqlZLRHDJyIUQc5HVMEwiOkjDIxmc3vI6z3cT9FfU0xqjojx2TkzyJ7pfsWK7EclAQgJQEICUAQBAEAQEOAPNYaT6g57pfo10N6iBv3W2Rg/l/iH4fgvPcQwdn9SC5Fdk4+31RNTVOQjY9ApQ2ssf5kMjB33a7/aVZcLel+numSsR+soqyIslkYciyR7T7iVCui42ST9yPNaSaPuWhlZG2Z0bmxPya8jqlJUzjBTa5PuZdcktWuRjLkaGRS0MsoeYo3PEY1nloyaF1hTOabitdDeMJS6I8LX2b9i0S1ZqkbPpv1BRQ+lFTjW99h/tKtOJvRQh7Il5XLavsauqkhmx6C13RVQjJ6s7Sz/EM2/Me9WfC7tlu19H/JKxJ7Z6e50sL0xaBZAQHP8AyjVmtLDCDkxpkcPxOyH7A+KqeIz9Siux6PglOkZWe/I09VpehAbe5n9pULC3OrpBqlvpSM/qAPeFY6fEUrT5kUKbwsl6/JLuYGjOMw0bZnPic6c5Rmwy/Cb9nPauWNdGrXcuZJzsSzJlHbL09zErNI6uVxcZ3t4NjOo0cslznlWyeup3q4djwWm3X9SxwTS+WJ2rUF00JyN7F7e7iORXanNlF6T5oi5XCoTW6rk/8FfFRfbax7aZhZG9+tmMo2b3H98lyVautah0JLu+Fx07Xq/5MzS+uYXR0kP8KlbqXGwvtY+GzvJXTLsTarj0Rw4ZRJJ3T6y/g19jSSGgElxAAG0k7AoaTb0RaSkorV9jqmi+CikhAIvK+xldz3NHIK+xqFVD7njs7LeRZr2XQulJIRKAIAgIQEoCEAQEoCEAQBAQ9gcCCAQQQQdhB3LDSa0ZhrU5LplRxUExBkY2J4LowXDWHFttuS81lcPnGzStaplZbjyU/SuRXYRiYDo6iFwdqPDgcxexzBv4e9REp49q3LRo5c656vsbPpXQiZrcQp+tFK0dKBtY8ZXPwPdzU3OpVi8evo+p2yIbv6ke4wHSZjIvs1WzpIbarXWDrN9Vzd47lnFzoxh4Vq1RmrISjsn0M00mDHr9LYbdTXkHuta67eHgPnuN3HHfPU8cT0nhiiNPQM1GkEGTV1QL7SAcyeZWl2fXCHh0L8mtmRGMdsEV+iWFCR/2mXq09P1y52xzhmB7tq4YOPul4s/lRpj16vdLoiuxzETU1Ek25xswcGDIfX3qLlX+NY5nK2e+bZgKMcj0p5jG9r29pjmvHe03HwW9cnCSkuxtF6NM7NTTCRjHt2Pa1w7iLr2kJKUVJF3F6rU9FuZIKGGcj0mqelrJ37tcsb3N6vyXnsme61s9rgV+HjxRWLgTAgMnDq6SnkEsTtVw8HD1SN4XSu2VctYnG+iF0NkzZXy0GI9Z5+x1J2uy6N5432H9iputN/X0sqFHKw/l9UDwfoTUHOOWnkbudruHyK0eDPs0dVxiv+6LR9M0RbF1qurhiaNoYdZx7r2+BWVhqPOySMS4rKfKmDb+5812PxQxmnw9hY05Pnd2392/3lYnkxhHZUvyKcCy2fi5L1+xrCglylpyNs0AwrpJXVDxdkOTOchG33D4qwwKd0t77FJxjJ2QVUer6/odECuDzRKAIAgCAIAgCAICEBKAqsY0jo6MXqamKL8LnXee5gzPggNCxvyz0zLto6eWd26SUiGP3DNx8AgNCxrylYpVXHTinYb9Smb0eXAuN3fugMTRTRKrxaV7mX6NtzLUylxGtbJgJzc4/ttPPWTaT06mH05GzUtG2Boia3VDCQQdt99+a8dfZOybc+pSzk5SbZd4Djj6RxFukhf/ABInbDzHArti5cqHp1T7G9Vzhy7Fs/CaGs69LOKeQ5mGXIA8uHuupbx8e/1VS0fszs667ecXozx8yar16cjj0jv+K5+V2+6MfCS90erNH6Sm69bVMeR/JizLjwO/4LosOmn1XT1+yMqiuHOb1K/HcfNQ0QxM6GmZbVjFgXW2F1vgo+Vm+ItkFpE5237vSuSKRV5HCAIZOoaFVPSUUQ3xl0Z9xy/Yheq4dZvoX25FrjS3Vl+p5IPKpk1GPefRY53gLrEnomzaEd0kjirnEkk7SST3nNeak9Xqe8gtsUj5WDYIAgCA+mvI2EjuJCzufuauEX1RBz2596w3r1MpJdAhkIDrmjlB9npYo7dbVDn/AJ3Zn6e5eior2VpHiMy523Sl/wDaFouxGCAIAgCAIAgCApsa0qoaIf3mqijd6mtryHuY25QGg415aIW3bRUr5TukncImd4aLk++yA0LGvKHilXcOqTCw3+7ph0Lbd+bj4oDVXkklxJLjtJJJPeUBCA3Tyf6Ay4m8Sya0NE13Wk2OmI2sj+bt3egP0Bh2HxU0TIII2xxRizWNFgPqeaA51ptQdDVuIFmzASDv2OHjn715biVOy5tdHzKrKhtn+pQKuI4QwffSO2azrcNYrffL3Ztqz4stWzBKwYIQBAEBvXk3n6lRHwdG8f4gQf8ASFf8Gn6ZRLHDlyaN1V2TSs0lk1aOpP8A+Lx4i3zXHIelUn9iThx3XwX3OQrzp7cIAgCAIAgCAlAZ2BU3S1VPHudKzW/KDc/sCu1EN1kV9yLm2eHRKX2OwheiPEkoAgCAIAgPGsqo4Y3yyvbHGwFz3vNmtA3koDlmkHlmjY5zKCn6W2QnnJYw8wwZkd5CA59jWnmJ1lxLVvYw/wAqD7ln+XrH3koDW95O85k7yUAQBAQgOi+TjycOri2qrA6OjuCyPsvqfm1nPad3FAd1p4GRsbHG1rGMaGsY0ANa0bAAEB6oDUPKNTXhhl3skLD+Vw+rR4qn4vDWCl7P+SHmR9KZz9edK0IAgCAIAgCAIDa/J1Japlb60N/0uH1Kt+Dy0ta+xMw36mjoa9EWRSaautQT8+jHi9qjZj0pZO4YtcqJypUKPZBAEAQBAEAQEoDYdA49auafVjld+1vmpmCtbSr4vLTH092dOV4eTCAIAgJQEIDhXlo0nfPV/YI3EQUuqZQDlJORfrcQ0EW5k8kBS6D6A1GK60geIKZjtUzOaXF7htaxu+287EBtWMeRd7Iy6kq+lkaCeimYGa/IOByPeEBymaJzHOY9pa9jnMe1wsWuBsWkcQQgPlAQgOs+TbyZmTUrMRZZmToaVwzfwfKOHBvjwQHZGtAAAFgMgBkAOCAlALICj01j1qGb8Oo7wcFB4jHXHkR8la1s5avJlSEMhDAQBDIQBAEMGx6BOtWjnFKPgfkrPhX/AHH4ZKxH/UOlr05aGFjOHCqhdC5xaHlpJba4sQd/cuV1SsjtZ3x73RYrF2Nb8wYfby/pYofl0Pcs/O7fpQ8wYvby/pYnl0Pced2/Sh5gxe3l/SxPLoe7Hndv0oeYMXt5f0sTy6Pux53b9KHmDD7eX9LE8uj7sed2/Sh5gxe3l/SxPLoe487t+lDzBi9vL+lieXQ9x53b9KHmDF7eX9LE8uh7jzu36UWOB6LR0kplbK95LCyzg0CxIN8u5dqMSNUtUyLl8Rnkw2SWhsClleSgIQEoAgCA/K2lpd/aNdr31vtdTe/5zb9rID9CeTcRjCKDo7av2dpNvaXOvfnrXQGyoD83+VgRjGavo7fyS+3tOjbre/YgNSY0uIa0FznENa1oJLicgABtKA7X5N/JoKfUrMQYHT5Ohp3WLYN4c/i/lu70B1JAEAQBAYmKUQqIZIXEtEjdUkWuM/6LldUrYOD7ms4qUWma15gw+3l8GfRVfk9f1Mi/Bx9x5gw+3l8GfRY8nh9THwa9x5gw+3l8GfRZ8mh9THwa9x5gw+3l8GfRY8nh9THwa9x5gw+3l8GfRPJ4fUx8GvceYMPt5fBn0TyeH1MfBx9x5gw+3l8GfRPJ4fUx8GvcjzBh9vL4M+iz5PD6mPg4+5nYNonHSzCZssjiA4WcG2zFty743D4UT3ps3qxlCW7U2NWLJJCyCUAQEICUBCAlAQgCAIAgCAIAgCAIDg3ln0afT1hrmNJp6u2u4DKOoAsQeGsACOd0BW6B+UGfCwYSz7RSucX9GXar43HaWHnwQG2Yx5aQ6Ito6R7JSLB9Q5haw8Q1pOt7yEBylrZqqewEk9RUSE2A1nySONyfElAd28nXk7jw8NqakNlrSLj0mU9x2WcXbi7wQHQEAQBAEAQBAEAQBAEAQBAEAQBAEBCAIAgCAIAgCAIAgCAIAgCAIAgCAx6+iiqI3wzMbLFINV7Hi7XBAcqx7yMAuLqGpDGnMQ1Ac4N5CRudu8FAVVJ5GK1zvvamljZvLOlldbkC1vxQHS9DtB6TCwXRAyzuFn1Eltcjg0bGjkEBtCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAhAEAQBAEAQBAEAQBAEAQBAEAQEoCEBKAIAgCAIAgCAIAgCAIAgCAIAgCAIAgP/Z")

st.title ("SENTIMENT ANALYSIS SYSTEM") #setting the title
choices=st.sidebar.selectbox("My Menu",("Home","Analysis","Results")) #sidebar menu
    # Home page content
if(choices=="Home"):
      # Displaying an image
    st.image("https://cdn.prod.website-files.com/5ebc4dae09fa971fce3e85a5/6132238846b8a0e52e357485_Screen%20Recording%202021-09-03%20at%2010.29.36.gif", width=600)
    # Description of the application
    st.write("1.) It is a Natural Language Processing Application which can analyze the sentiment on a text data.")  # Description of the application
    st.write("2.) This Application predicts the sentiment into 3 categories Positive, Negative and Neutral.")
    st.write("3.) This Application then visualizes the results based on different factors such as gender, age etc.")
    # Analysis page content
elif(choices=="Analysis"):
     # User input for Google Sheet details
    gsid=st.text_input("Enter your google Sheet ID")
    r=st.text_input("Enter Range between first column and last column")
    c=st.text_input("Enther the column name that is to be analyzed")
    btn=st.button("Analyze")
    # When Analyze button is clicked
    if btn:
         # Check if credentials already exist in session state
        if 'cred' not in st.session_state:
            # Load credentials from the key.json file with access to Google Sheets
            f=InstalledAppFlow.from_client_secrets_file("key.json",["https://www.googleapis.com/auth/spreadsheets"])
            # Run the local server and store credentials
            st.session_state['cred']=f.run_local_server(port=0)
         # Initialize sentiment analysis model
        mymodel=SentimentIntensityAnalyzer()
         # Build the Google Sheets API service
        service=build("Sheets","v4",credentials=st.session_state['cred']).spreadsheets().values()
             # Fetch data from Google Sheet
        k=service.get(spreadsheetId=gsid, range=r).execute()
        d=k['values']
        df=pd.DataFrame(data=d[1:],columns=d[0]) # Convert sheet data to DataFrame
        l=[]
        for i in range(0,len(df)):# Iterate over each row and perform sentiment analysis
            t=df._get_value(i,c) # Extract text from specified column
            pred=mymodel.polarity_scores(t) # Predict sentiment
            if(pred['compound']>0.5):  # Classify sentiment based on compound score
                l.append("Posittive")
            elif(pred["compound"]<-0.5):
                l.append("Negative")
            else:
                l.append("Neutral")
        # Add the sentiment results to the DataFrame
        df['Sentiment']=l
        df.to_csv("results.csv",index=False)# Save the results to a CSV file
        st.subheader("The Analysis results are saved by the name of a results.csv")# Show message to user
elif(choices=="Results"): # Results section with visualizations
    df=pd.read_csv("results.csv")   # Load the saved results
     # Allow user to choose type of visualization
    choice2=st.selectbox("Choose Visualisation",("None","Pie Chart","Histogram","Box Plot"))
    st.dataframe(df)   # Display the full DataFrame
    
    if(choice2=="Pie Chart"):   # Pie Chart for Sentiment Distribution
        posper = (len(df[df['Sentiment'] == 'Positive']) / len(df)) * 100
        negper = (len(df[df['Sentiment'] == 'Negative']) / len(df)) * 100
        neuper = (len(df[df['Sentiment'] == 'Neutral']) / len(df)) * 100
        # Create pie chart
        fig=px.pie(values=[posper,negper,neuper],names=['Positive','Negative','Neutral'],title='Sentiment Distribution')
        st.plotly_chart(fig)
        
    #user selects Histogram
    elif(choice2=="Histogram"):
        p=st.selectbox("Choose Column",df.columns)  # Histogram for selected column
        if p: # Create histogram with sentiment coloring
            fig=px.histogram(x=df[p],color=df['Sentiment'],title='Distribution of Sentiments with Respect to Selected Column"')
            fig.update_layout(xaxis_title=p, yaxis_title="Count") 
            st.plotly_chart(fig)#Display the histogram
            
      # Box Plot for selected column
    elif (choice2 == "Box Plot"):
          # Display a dropdown for the user to choose a column from the dataset
        l = st.selectbox("Choose Column", df.columns)
        if l:
            fig = px.box(x=df[l], color=df['Sentiment'],title='Sentiment-Based Spread of {Selected Column}')
            fig.update_layout(xaxis_title="Sentiment",yaxis_title=l)
            st.plotly_chart(fig)   # Display the box plot

            

            
    

