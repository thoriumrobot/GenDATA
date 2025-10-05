/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
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
