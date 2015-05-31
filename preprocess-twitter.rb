# Ruby 2.0
# Reads stdin: ruby -n preprocess-twitter.rb
#
# Script for preprocessing tweets by Romain Paulus
# with small modifications by Jeffrey Pennington
require 'csv'
def tokenize input

	# Different regex parts for smiley faces
	eyes = "[8:=;]"
	nose = "['`\-]?"

	input = input
		.gsub(/https?:\/\/\S+\b|www\.(\w+\.)+\S*/,"<URL>")
		.gsub("/"," / ") # Force splitting words appended with slashes (once we tokenized the URLs, of course)
		.gsub(/@\w+/, "<USER>")
		.gsub(/#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i, "<SMILE>")
		.gsub(/#{eyes}#{nose}p+/i, "<LOLFACE>")
		.gsub(/#{eyes}#{nose}\(+|\)+#{nose}#{eyes}/, "<SADFACE>")
		.gsub(/#{eyes}#{nose}[\/|l*]/, "<NEUTRALFACE>")
		.gsub(/<3/,"<HEART>")
		.gsub(/[-+]?[.\d]*[\d]+[:,.\d]*/, "<NUMBER>")
		.gsub(/#\S+/){ |hashtag| # Split hashtags on uppercase letters
			# TODO: also split hashtags with lowercase letters (requires more work to detect splits...)

			hashtag_body = hashtag[1..-1]
			if hashtag_body.upcase == hashtag_body
				result = "<HASHTAG> #{hashtag_body} <ALLCAPS>"
			else
				result = (["<HASHTAG>"] + hashtag_body.split(/(?=[A-Z])/)).join(" ")
			end
			result
		} 
		.gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			"#{$~[1]} <REPEAT>"
		}
		.gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
			# TODO: determine if the end letter should be repeated once or twice (use lexicon/dict)
			$~[1] + $~[2] + " <ELONG>"
		}
		.gsub(/ ([^a-z0-9()<>'`\-]){2,} /){ |word|
			"#{word.downcase} <ALLCAPS>"
		}

	return input.downcase
end

filename = "tweeti.b.dev"
CSV.open("#{filename}.preprocessed", "wb", :col_sep=>"\t",quote_char:"\x00") do |csv|
	CSV.foreach(filename ,{ :encoding=>"ISO8859-1", :col_sep=>"\t",quote_char:"\x00"}) do |row|
		puts row[3]
		row[3] = tokenize row[3]
		csv << row
	end
end

# CSV.open("#{filename}.preprocessed","wb") do |csv|
# 	csv << ["hello","a"]
# end

# CSV.foreach(filename ,{ :encoding=>"ISO8859-1", :col_sep=>"\t",quote_char:"\x0"}) do |row|
# 	puts row[3]
# 	puts tokenize row[3]
# end


# text = File.read("tweeti.b.dev")
# # puts text ISO8859-1 :encoding=>"utf-8"
# CSV.parse(text, {headers: false,:encoding=>"utf-8",col_sep:"\t",quote_char:"\x0"}) do |row|
# 		puts tokenize row[3]
#   # use row here...
# end

# File.foreach("tweeti.b.dev") do |csv_line|

#   row = CSV.parse(csv_line.gsub('\"', '""'), {col_sep: '\t'}).first
# 	puts row

# end

# CSV.open("path/to/file.csv", "wb") do |csv|
#   csv << ["row", "of", "CSV", "data"]
#   csv << ["another", "row"]
#   # ...
# end

# puts tokenize($_)
