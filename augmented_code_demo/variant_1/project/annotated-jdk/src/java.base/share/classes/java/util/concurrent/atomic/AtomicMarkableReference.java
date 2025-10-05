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
package java.util.concurrent.atomic;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AtomicMarkableReference<V> {

    @Positive
    private static class Pair<T> {

    @Positive
        static <T> Pair<T> of(T reference, boolean mark);
    @Positive
    }

    @Positive
    public AtomicMarkableReference(V initialRef, boolean initialMark) {
    @Positive
    }

    @Positive
    public V getReference();

    @Positive
    public boolean isMarked();

    @Positive
    public V get(boolean[] markHolder);

    @Positive
    public boolean weakCompareAndSet(V expectedReference, V newReference, boolean expectedMark, boolean newMark);

    @Positive
    public boolean compareAndSet(V expectedReference, V newReference, boolean expectedMark, boolean newMark);

    @Positive
    public void set(V newReference, boolean newMark);

    @Positive
    public boolean attemptMark(V expectedReference, boolean newMark);
    @Positive
}

// CFWR semantic augmentation - variant 1
