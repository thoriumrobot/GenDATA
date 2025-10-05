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
package java.util;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.KeyFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;

@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
@AnnotatedFor({ "lock", "nullness" })
public interface NavigableMap<K, V> extends SortedMap<K, V> {

    Map.@Nullable Entry<K, V> lowerEntry(K key);

    @Nullable
    K lowerKey(K key);

    Map.@Nullable Entry<K, V> floorEntry(K key);

    @Nullable
    K floorKey(K key);

    Map.@Nullable Entry<K, V> ceilingEntry(K key);

    @Nullable
    K ceilingKey(K key);

    Map.@Nullable Entry<K, V> higherEntry(K key);

    @Nullable
    K higherKey(K key);

    Map.@Nullable Entry<K, V> firstEntry();

    Map.@Nullable Entry<K, V> lastEntry();

    Map.@Nullable Entry<K, V> pollFirstEntry(@GuardSatisfied NavigableMap<K, V> this);

    Map.@Nullable Entry<K, V> pollLastEntry(@GuardSatisfied NavigableMap<K, V> this);

    @SideEffectFree
    NavigableMap<K, V> descendingMap();

    @SideEffectFree
    NavigableSet<@KeyFor({ "this" }) K> navigableKeySet();

    @SideEffectFree
    NavigableSet<@KeyFor({ "this" }) K> descendingKeySet();

    @SideEffectFree
    NavigableMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @SideEffectFree
    NavigableMap<K, V> headMap(K toKey, boolean inclusive);

    @SideEffectFree
    NavigableMap<K, V> tailMap(K fromKey, boolean inclusive);

    @SideEffectFree
    SortedMap<K, V> subMap(K fromKey, K toKey);

    @SideEffectFree
    SortedMap<K, V> headMap(K toKey);

    @SideEffectFree
    SortedMap<K, V> tailMap(K fromKey);
}
