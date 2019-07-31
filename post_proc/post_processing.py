def post_processing(word):

	# change to origin english letters
	uzi_arnon_symbols = "".maketrans({"h": "H", "w": "v", "ḥ": "h", "ṭ": "t", "ç": "ts", "q": "k", "š": "sh"})
	word = word.translate(uzi_arnon_symbols)

	# handling letters א-ה-ע-ו-י and remove rhe symbol *
	idxs_to_remove = []
	y_letters = [i for i, let in enumerate(word) if let == "y"]
	for idx in y_letters:
		if (word[idx+1] == "*" and
			(word[idx-1] == "i" or word[idx-2] == "y" or
			(idx+3 < len(word) and word[idx+2] == "v" and word[idx+3] == "*"))) \
			or word[idx+1] == "i":
				idxs_to_remove.append(idx)

	v_letters = [i for i, let in enumerate(word) if let == "v" and i != 0]
	for idx in v_letters:
		if word[idx + 1] == "*":
			if word[idx - 1] == "o" or word[idx - 1] == "u":
				idxs_to_remove.append(idx)
				continue
			if word[idx-2] == "v" or (idx+2 < len(word) and word[idx+2] == "v"):
				idxs_to_remove.append(idx)

	idxs_to_remove.sort()
	removed = 0
	for idx in idxs_to_remove:
		idx -= removed
		word = word[0:idx] + word[idx+1:]
		removed += 1

	for letter in "ˀHˁ*":
		word = word.replace(letter, "")


	#exceptions

	if word == "i":
		word = "yi"

	'''word = list(word)
	if word[len(word)-1] == "h":
		word.insert(len(word)-1, "a")
	word = "".join(word)'''

	return (word)




if __name__ == "__main__":
    print(post_processing ("ko-ḥ*"))