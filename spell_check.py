import jamspell
from spellchecker import SpellChecker

def spell_correct_context(query_str):
    corrector = jamspell.TSpellCorrector()    # Create a corrector
    corrector.LoadLangModel('/home/malavikka/Desktop/AIR_proj/Information-Retrieval-System/en.bin')  
    list_of_words = get_list(query_str)
    #PRINTING THE CANDIDATES 
    # for i in range(len(list_of_words)):
        # print(list_of_words[i]+" -> ", corrector.GetCandidates(list_of_words, i))
    #print("Did you mean " + "'"+corrector.FixFragment(query_str)+ "'"+"?")
    return corrector.FixFragment(query_str)

def spell_correct(query_str):
    spell = SpellChecker()
    # find those words that may be misspelled
    list_of_words = get_list(query_str)
    new_str = ""
    for word in list_of_words:
    # Get the one `most likely` answer
        new_str += str((" "+spell.correction(word)))
    # Get a list of `likely` options
        print(word+' ->',spell.candidates(word))
    print("Did you mean "+"'"+new_str+"'"+"?")

def get_list(string):
    return string.split()

def main():
    print("With context")
    spell_correct_context("I am tge begt spell cherken")
    print("\n")
    print("Without context")
    spell_correct("I am tge begt spell cherken")

if __name__ == "__main__":
    main()
    
