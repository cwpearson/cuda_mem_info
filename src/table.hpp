#include <string>
#include <sstream>
#include <vector>

class Table {
private:
    std::vector<std::vector<std::string>> rows_;

public:
    void Cell(const std::string &s) {
        if (rows_.empty()) {
            rows_.push_back({});
        }
        rows_[rows_.size() - 1].push_back(s);
    }
    void EndRow() {
        rows_.push_back({});
    }

    std::string csv_str() {
        std::stringstream ss;

        for (const auto row : rows_) {
            for (size_t i = 0; i < row.size(); ++i) {
                ss << row[i];
                if (i + 1 < row.size()) {
                    ss << ",";
                }
            }
            ss << std::endl;
        }
        return ss.str();
    }
    std::string md_str() {
        return "";
    }
};