#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

#include <type_traits>
#include <utility>

// By default an object is not a container
template <typename T, typename = void> struct isContiguousContainer : std::false_type {};

// Checks if a given type is a container with a valid size() and data() method
// Critical for bypassing incorrect sizeof() compile check
template <typename T>
struct isContiguousContainer<T, std::void_t<decltype(std::declval<T>().size(), std::declval<T>().data())>>
    : std::true_type {};

#endif
