#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

#include <concepts>

template <typename T>
concept ContiguousContainer = requires(T t) {
    { t.data() } -> std::convertible_to<const void *>;
    { t.size() } -> std::convertible_to<size_t>;

    T::value_type;
};

#endif
