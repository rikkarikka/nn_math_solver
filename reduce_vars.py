#
def reduce_variables(input_txt, input_eqs, output_txt, output_eqs):
    input_txt_lst = []
    input_eqs_lst = []
    output_txt_lst = []
    output_eqs_lst = []

    with open(input_txt, 'r') as f:
        input_txt_lst = f.readlines()
    with open(input_eqs, 'r') as f:
        input_eqs_lst = f.readlines()
    open(output_txt, 'w').close()
    open(output_eqs, 'w').close()

    i = 0
    for input_txt_line in input_txt_lst:
        txt_words = input_txt_line.split()
        txt_vars = [word for word in txt_words if '_' in word]
        eqs_words = input_eqs_lst[i].split()
        eqs_vars = [word for word in eqs_words if '_' in word]

        # Create Dictionary
        replacements = {}
        j = 1
        for var in txt_vars:
            if var not in replacements.keys():
                if (var in eqs_vars) and ('VAR_' not in var):
                    replacements[var] = 'NUMBER_' + str(j)
                    j = j + 1
                else:
                    if 'NUMBER_' in var:
                        j = j + 1
                    replacements[var] = var
        final_txt_words = []
        output_txt_line = ''
        for word in txt_words:
            if word in replacements.keys():
                final_txt_words.append(replacements.get(word))
                output_txt_line = output_txt_line + replacements.get(word) + ' '
            else:
                final_txt_words.append(word)
                output_txt_line = output_txt_line + word + ' '
        output_txt_line = output_txt_line.strip()

        final_eqs_words = []
        output_eqs_line = ''
        for word in eqs_words:
            if word in replacements.keys():
                final_eqs_words.append(replacements.get(word))
                output_eqs_line = output_eqs_line + replacements.get(word) + ' '
            else:
                final_eqs_words.append(word)
                output_eqs_line = output_eqs_line + word + ' '
        output_eqs_line = output_eqs_line.strip()
        i = i + 1

        # Write results to file
        with open(output_txt, 'a') as f:
            f.writelines(output_txt_line + '\n')
        with open(output_eqs, 'a') as f:
            f.writelines(output_eqs_line + '\n')

def main():
    print('Running fewer_vars.py...')
    reduce_variables('./data/train.txt', './data/train.eq', './data/train.red.txt', './data/train.red.eq')
    #reduce_variables('./data/val.txt', './data/val.eq', './data/val.red.txt', './data/val.red.eq')
    #reduce_variables('./data/test.txt', './data/test.eq', './data/test.red.txt', './data/test.red.eq')
    print('fewer_vars.py complete...')

if __name__ == "__main__":
    main()
