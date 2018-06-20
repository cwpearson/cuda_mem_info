#ifndef ELEMENT_HPP
#define ELEMENT_HPP

class Element {
public:
    virtual std::string md_str() const = 0;
    virtual std::string ascii_str() const = 0;
};

#endif