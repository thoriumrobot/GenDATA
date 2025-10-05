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
package java.util.concurrent;

    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.util.NavigableMap;
    @Positive
import java.util.NavigableSet;

    @Positive
public interface ConcurrentNavigableMap<K, V> extends ConcurrentMap<K, V>, NavigableMap<K, V> {

    @Positive
    @SideEffectFree
    @Positive
    ConcurrentNavigableMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @Positive
    @SideEffectFree
    @Positive
    ConcurrentNavigableMap<K, V> headMap(K toKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    ConcurrentNavigableMap<K, V> tailMap(K fromKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    ConcurrentNavigableMap<K, V> subMap(K fromKey, K toKey);

    @Positive
    @SideEffectFree
    @Positive
    ConcurrentNavigableMap<K, V> headMap(K toKey);

    @Positive
    @SideEffectFree
    @Positive
    ConcurrentNavigableMap<K, V> tailMap(K fromKey);

    @Positive
    @SideEffectFree
    @Positive
    ConcurrentNavigableMap<K, V> descendingMap();

    @Positive
    @SideEffectFree
    @Positive
    NavigableSet<K> navigableKeySet();

    @Positive
    NavigableSet<K> keySet();

    @Positive
    @SideEffectFree
    @Positive
    NavigableSet<K> descendingKeySet();
    @Positive
}
