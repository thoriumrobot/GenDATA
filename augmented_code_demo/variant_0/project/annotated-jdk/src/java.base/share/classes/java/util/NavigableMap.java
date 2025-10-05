/*
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.util;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;

    @Positive
@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public interface NavigableMap<K, V> extends SortedMap<K, V> {

    @Positive
    Map.@Nullable Entry<K, V> lowerEntry(K key);

    @Positive
    @Nullable
    @Positive
    K lowerKey(K key);

    @Positive
    Map.@Nullable Entry<K, V> floorEntry(K key);

    @Positive
    @Nullable
    @Positive
    K floorKey(K key);

    @Positive
    Map.@Nullable Entry<K, V> ceilingEntry(K key);

    @Positive
    @Nullable
    @Positive
    K ceilingKey(K key);

    @Positive
    Map.@Nullable Entry<K, V> higherEntry(K key);

    @Positive
    @Nullable
    @Positive
    K higherKey(K key);

    @Positive
    Map.@Nullable Entry<K, V> firstEntry();

    @Positive
    Map.@Nullable Entry<K, V> lastEntry();

    @Positive
    Map.@Nullable Entry<K, V> pollFirstEntry(@GuardSatisfied NavigableMap<K, V> this);

    @Positive
    Map.@Nullable Entry<K, V> pollLastEntry(@GuardSatisfied NavigableMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    NavigableMap<K, V> descendingMap();

    @Positive
    @SideEffectFree
    @Positive
    NavigableSet<@KeyFor({ "this" }) K> navigableKeySet();

    @Positive
    @SideEffectFree
    @Positive
    NavigableSet<@KeyFor({ "this" }) K> descendingKeySet();

    @Positive
    @SideEffectFree
    @Positive
    NavigableMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @Positive
    @SideEffectFree
    @Positive
    NavigableMap<K, V> headMap(K toKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    NavigableMap<K, V> tailMap(K fromKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    SortedMap<K, V> subMap(K fromKey, K toKey);

    @Positive
    @SideEffectFree
    @Positive
    SortedMap<K, V> headMap(K toKey);

    @Positive
    @SideEffectFree
    @Positive
    SortedMap<K, V> tailMap(K fromKey);
    @Positive
}

// CFWR semantic augmentation - variant 0
