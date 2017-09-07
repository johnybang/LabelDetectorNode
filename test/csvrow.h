#ifndef CSVROW_H_
#define CSVROW_H_

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace jyb {

// Pilfered from:
// https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
class CSVRow
{
public:
	std::string const& operator[](std::size_t index) const
	{
		return m_data[index];
	}
	std::size_t size() const
	{
		return m_data.size();
	}
	void readNextRow(std::istream& str)
	{
		std::string					line;
		std::getline(str, line);

		std::stringstream		lineStream(line);
		std::string					cell;

		m_data.clear();
		while(std::getline(lineStream, cell, ','))
		{
			m_data.push_back(cell);
		}
		// This checks for a trailing comma with no data after it.
		if (!lineStream && cell.empty())
		{
			// If there was a trailing comma then add an empty element.
			m_data.push_back("");
		}
	}
private:
	std::vector<std::string>		m_data;
};

} // namespace jyb

std::istream& operator>>(std::istream& str, jyb::CSVRow& data)
{
	data.readNextRow(str);
	return str;
}

#endif // CSVROW_H_
