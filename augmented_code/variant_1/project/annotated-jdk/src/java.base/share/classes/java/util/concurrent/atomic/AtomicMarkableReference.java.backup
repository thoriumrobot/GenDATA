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
package java.util.concurrent.atomic;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class AtomicMarkableReference<V> {

    private static class Pair<T> {

        static <T> Pair<T> of(T reference, boolean mark);
    }

    public AtomicMarkableReference(V initialRef, boolean initialMark) {
    }

    public V getReference();

    public boolean isMarked();

    public V get(boolean[] markHolder);

    public boolean weakCompareAndSet(V expectedReference, V newReference, boolean expectedMark, boolean newMark);

    public boolean compareAndSet(V expectedReference, V newReference, boolean expectedMark, boolean newMark);

    public void set(V newReference, boolean newMark);

    public boolean attemptMark(V expectedReference, boolean newMark);
}
