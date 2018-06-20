#include "table.hpp"

std::string Table::md_str() const {
    std::stringstream ss;
    const size_t num_rows = NumRows();
    const size_t num_cols = NumCols();

    ss << "**" << title_ << "**" << std::endl;

    // print headers
    ss << header_.md_str() << std::endl;

    // print title break
    ss << "|";
    for (size_t i = 0; i < num_cols; ++i) {
        ss << "-|";
    }
    ss << std::endl;

    // print rows
    for (size_t rowIdx = 0; rowIdx < num_rows; ++rowIdx) {

        const auto &row = rows_[rowIdx];
        ss << row.md_str(num_cols) << std::endl;

    }

    return ss.str();
}

std::string Table::ascii_str() const {
    std::stringstream ss;
    const size_t num_cols = NumCols();

    // Compute display width of each column
    std::vector<size_t> col_display_widths;
    for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
        const size_t display_width = ColWidth(col_idx) + 2;
        col_display_widths.push_back(display_width);
    }

    auto divider = [&]() {
        ss << "+";
        for (auto width : col_display_widths) {
            for (size_t i = 0; i < width; ++i) {
                ss << "-";
            }
            ss << "+";
        }
        ss << "\n";
    };


    ss << title_ << std::endl;
    divider();
    ss << header_.shell_str(num_cols, col_display_widths) << std::endl;
    divider();
    for (const auto &row : rows_) {
        ss << row.shell_str(num_cols, col_display_widths) << std::endl;
    }
    divider();

    return ss.str();
}