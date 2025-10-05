/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.common.aliasing.qual.NonLeaked;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import org.checkerframework.framework.qual.Covariant;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Function;
    @Positive
import java.io.Serializable;

    @Positive
@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index", "aliasing", "nonempty" })
    @Positive
public interface Map<K, V> {

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    int size(@GuardSatisfied Map<K, V> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(if (false, expression = "this")
    @Positive
    boolean isEmpty(@GuardSatisfied Map<K, V> this);

    @Positive
    @CFComment("nullness: key is not @Nullable because this map might not permit null values")
    @Positive
    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = { "this" })
    @Positive
    @Pure
    @Positive
    boolean containsKey(@GuardSatisfied Map<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @EnsuresNonEmptyIf(result = true, expression = { "this" })
    @Positive
    @Pure
    @Positive
    boolean containsValue(@GuardSatisfied Map<K, V> this, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    V get(@GuardSatisfied Map<K, V> this, @UnknownSignedness @GuardSatisfied Object key);

    @Positive
    @ReleasesNoLocks
    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    @Nullable
    @Positive
    V put(@GuardSatisfied Map<K, V> this, K key, V value);

    @Positive
    @CFComment("nullness: key is not @Nullable because this map might not permit null values")
    @Positive
    @Nullable
    @Positive
    V remove(@GuardSatisfied Map<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    void putAll(@GuardSatisfied Map<K, V> this, Map<) {
            result = extends K, ? extends V> m);

    @Positive
    void clear(@GuardSatisfied Map<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNonEmpty
    @Positive
    Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied @PolyNonEmpty Map<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNonEmpty
    @Positive
    Collection<V> values(@GuardSatisfied @PolyNonEmpty Map<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNonEmpty
    @Positive
    Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied @PolyNonEmpty Map<K, V> this);

    @Positive
    @Covariant({ 0 })
    @Positive
    interface Entry<K, V> {

    @Positive
        @Pure
    @Positive
        K getKey(Map.@GuardSatisfied Entry<K, V> this);

    @Positive
        @Pure
    @Positive
        V getValue(Map.@GuardSatisfied Entry<K, V> this);

    @Positive
        V setValue(Map.@GuardSatisfied Entry<K, V> this, V value);

    @Positive
        @Pure
    @Positive
        boolean equals(Map.@GuardSatisfied Entry<K, V> this, @GuardSatisfied @Nullable Object o);

    @Positive
        @Pure
    @Positive
        int hashCode(Map.@GuardSatisfied Entry<K, V> this);

    @Positive
        @Pure
    @Positive
        public static <K extends Comparable<? super K>, V> Comparator<Map.Entry<K, V>> comparingByKey();

    @Positive
        @Pure
    @Positive
        public static <K, V extends Comparable<? super V>> Comparator<Map.Entry<K, V>> comparingByValue();

    @Positive
        @Pure
    @Positive
        public static <K, V> Comparator<Map.Entry<K, V>> comparingByKey(Comparator<? super K> cmp);

    @Positive
        @Pure
    @Positive
        public static <K, V> Comparator<Map.Entry<K, V>> comparingByValue(Comparator<? super V> cmp);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public static <K extends @NonNull Object, V extends @NonNull Object> Map.Entry<K, V> copyOf(Map.Entry<? extends K, ? extends V> e);
    @Positive
    }

    @Positive
    boolean equals(@GuardSatisfied Map<K, V> this, @GuardSatisfied @Nullable Object o);

    @Positive
    int hashCode(@GuardSatisfied Map<K, V> this);

    @Positive
    @Pure
    @Positive
    default V getOrDefault(@GuardSatisfied @UnknownSignedness Object key, V defaultValue);

    @Positive
    default void forEach(@NonLeaked BiConsumer<? super K, ? super V> action);

    @Positive
    default void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    default V putIfAbsent(K key, V value);

    @Positive
    @CFComment("nullness;
        } else {
            result = key and value are not @Nullable because this map might not permit null values")
    @Positive
    default boolean remove(@GuardSatisfied @UnknownSignedness Object key, @GuardSatisfied @UnknownSignedness Object value);
        }

    @Positive
    default boolean replace(K key, V oldValue, V newValue);

    @Positive
    @Nullable
    @Positive
    default V replace(K key, V value);

    @Positive
    @PolyNull
    @Positive
    default V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
    @Nullable
    @Positive
    default V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @Nullable V> remappingFunction);

    @Positive
    @Nullable
    @Positive
    default V compute(K key, BiFunction<? super K, ? super @Nullable V, ? extends @Nullable V> remappingFunction);

    @Positive
    @Nullable
    @Positive
    default V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @Nullable V> remappingFunction);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <K, V> Map<K, V> of();

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4, K k5, V v5);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4, K k5, V v5, K k6, V v6);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4, K k5, V v5, K k6, V v6, K k7, V v7);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4, K k5, V v5, K k6, V v6, K k7, V v7, K k8, V v8);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4, K k5, V v5, K k6, V v6, K k7, V v7, K k8, V v8, K k9, V v9);

    @Positive
    @NonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> of(K k1, V v1, K k2, V v2, K k3, V v3, K k4, V v4, K k5, V v5, K k6, V v6, K k7, V v7, K k8, V v8, K k9, V v9, K k10, V v10);

    @Positive
    @SafeVarargs
    @Positive
    @SuppressWarnings("varargs")
    @Positive
    @PolyNonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> ofEntries(Entry<? extends K, ? extends V>@PolyNonEmpty ... entries);

    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Entry<K, V> entry(@NonNull K k, @NonNull V v);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    @PolyNonEmpty
    @Positive
    static <K extends @NonNull Object, V extends @NonNull Object> Map<K, V> copyOf(@PolyNonEmpty Map<? extends K, ? extends V> map);
    @Positive
}
