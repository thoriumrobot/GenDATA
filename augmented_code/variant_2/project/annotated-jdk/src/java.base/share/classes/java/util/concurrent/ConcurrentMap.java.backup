/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util.concurrent;

import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;

@AnnotatedFor({ "nullness" })
public interface ConcurrentMap<K extends @NonNull Object, V extends @NonNull Object> extends Map<K, V> {

    @Override
    @Pure
    default V getOrDefault(Object key, V defaultValue);

    @Override
    default void forEach(BiConsumer<? super K, ? super V> action);

    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Nullable
    V putIfAbsent(K key, V value);

    boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

    boolean replace(K key, V oldValue, V newValue);

    @Nullable
    V replace(K key, V value);

    @Override
    default void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Override
    @PolyNull
    default V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Override
    @PolyNull
    default V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Override
    @PolyNull
    default V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Override
    @PolyNull
    default V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);
}
