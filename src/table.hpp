#include <string>
#include <sstream>
#include <vector>

#include <string>
#include <cstdarg>
#include <vector>
#include <string>
#include <iomanip>

std::string
vformat (const char *fmt, va_list ap)
{
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.
    size_t size = 1024;
    char buf[size];

    // Try to vsnprintf into our buffer.
    va_list apcopy;
    va_copy (apcopy, ap);
    int needed = vsnprintf (&buf[0], size, fmt, ap);
    // NB. On Windows, vsnprintf returns -1 if the string didn't fit the
    // buffer.  On Linux & OSX, it returns the length it would have needed.

    if (needed <= size && needed >= 0) {
        // It fit fine the first time, we're done.
        return std::string (&buf[0]);
    } else {
        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So do a malloc of the right size and try again.
        // This doesn't happen very often if we chose our initial size
        // well.
        std::vector <char> buf;
        size = needed;
        buf.resize (size);
        needed = vsnprintf (&buf[0], size, fmt, apcopy);
        return std::string (&buf[0]);
    }
}

std::string
format (const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat(fmt, ap);
    va_end (ap);
    return buf;
}


class Row {
private:
    typedef std::vector<std::string> container;
    container row_;
public:
    container::iterator begin() {
        return row_.begin();
    }

    container::iterator end() {
        return row_.end();
    }

    void resize(size_t n, container::value_type val = container::value_type()) {
        row_.resize(n, val);
    }

    size_t size() const {
        return row_.size();
    }

    void push_back(const std::string &s) {
        row_.push_back(s);
    }

    const std::string &operator[](const size_t i) const {
        return row_[i];
    }

    std::string &operator[](const size_t i) {
        return row_[i];
    }

    std::string csv_str(size_t pad = 0) const {
        std::stringstream ss;
        const size_t size = std::max(pad, row_.size());
        for (size_t i = 0; i < size; ++i) {
            if (i < row_.size()) {
                ss << row_[i];
            }
            if (i + 1 < size) {
                ss << ",";
            }
        }
        return ss.str();
    }

    std::string md_str (size_t pad = 0) const {
        std::stringstream ss;
        const size_t size = std::max(pad, row_.size());
        for (size_t i = 0; i < size; ++i) {
            if (i < row_.size()) {
                ss << row_[i];
            }
            ss << "|";
        }
        return ss.str();
    }

    std::string shell_str(const size_t pad = 0, const std::vector<size_t> col_pads = {}) const {
        std::stringstream ss;
        const size_t num_cols = std::max(pad, row_.size());

        ss << "|";
        for (size_t i = 0; i < num_cols; ++i) {
            ss << " ";

            if ( i < col_pads[i]) {
                const size_t display_width = col_pads[i];
                ss << std::setw(display_width - 2);
                if (i < row_.size()) {
                    ss << row_[i];
                } else {
                    ss << "";
                }
                ss << std::setw(0);
            } else {

            }
            ss << " |";
        }
        return ss.str();
    }
};

class Table {
private:
    Row header_;
    std::vector<Row> rows_;
    std::string title_;

    size_t NumRows(const bool header=false) const {
        const size_t num = rows_.size();
        if (header) {
            return num + 1;
        } else {
            return num;
        }
    }

    size_t NumCols() const {
        size_t num_cols = header_.size();
        for (const auto &row : rows_) {
            num_cols = std::max(row.size(), num_cols);
        }
        return num_cols;
    }

    size_t ColWidth(const size_t col_idx) {
        size_t width = 0;

        if (col_idx < header_.size()) {
            width = header_[col_idx].size();
        }

        for (const auto &row : rows_) {
            if (col_idx < row.size()) {
                auto &cell = row[col_idx];
                width = std::max(cell.size(), width);
            }
        }        
        return width;
    }

public:
    void Cell(const std::string &s = "") {
        if (rows_.empty()) {
            NewRow();
        }
        rows_[rows_.size() - 1].push_back(s);
    }

    void Cellf(const char *format, ...) {
        va_list ap;
        va_start (ap, format);
        std::string buf = vformat(format, ap);
        va_end (ap);
        Cell(buf);
    }

    void Titlef(const char *format, ...) {
        va_list ap;
        va_start (ap, format);
        std::string buf = vformat(format, ap);
        va_end (ap);
        title_ = buf;
    }

    std::string &Header(const size_t col_idx) {
        if (col_idx >= header_.size()) {
            header_.resize(col_idx + 1);
        }
        return header_[col_idx];
    }

    void NewRow() {
        rows_.push_back(Row());
    }

    std::string csv_str() {
        std::stringstream ss;

        const auto num_cols = NumCols();

        for (const auto row : rows_) {
            ss << row.csv_str();
            ss << std::endl;
        }
        return ss.str();
    }

    std::string md_str() {
        std::stringstream ss;
        const size_t num_rows = NumRows();
        const size_t num_cols = NumCols();

        for (size_t rowIdx = 0; rowIdx < num_rows; ++rowIdx) {

            const auto &row = rows_[rowIdx];

            if (num_cols > 0) {
                ss << "|";
            }

            ss << row.md_str(num_cols);
            ss << "\n";

            // first row, print title break
            if (0 == rowIdx) {
                if (num_cols > 0) {
                    ss << "|";
                }
                for (size_t i = 0; i < num_cols; ++i) {
                    ss << "-|";
                }
            ss << "\n";
            }

        }

        return ss.str();
    }

    std::string shell_str() {
        std::stringstream ss;
        const size_t num_rows = NumRows();
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
};