/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.lang.reflect.Array;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.function.BinaryOperator;
    @Positive
import java.util.function.UnaryOperator;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AtomicReferenceArray<E> implements java.io.Serializable {

    @Positive
    public AtomicReferenceArray(int length) {
    @Positive
    }

    @Positive
    public AtomicReferenceArray(E[] array) {
    @Positive
    }

    @Positive
    public final int length();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public final E get(int i);

    @Positive
    public final void set(int i, E newValue);

    @Positive
    public final void lazySet(int i, E newValue);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public final E getAndSet(int i, E newValue);

    @Positive
    public final boolean compareAndSet(int i, E expectedValue, E newValue);

    @Positive
    @Deprecated()
    @Positive
    public final boolean weakCompareAndSet(int i, E expectedValue, E newValue);

    @Positive
    public final boolean weakCompareAndSetPlain(int i, E expectedValue, E newValue);

    @Positive
    public final E getAndUpdate(int i, UnaryOperator<E> updateFunction);

    @Positive
    public final E updateAndGet(int i, UnaryOperator<E> updateFunction);

    @Positive
    public final E getAndAccumulate(int i, E x, BinaryOperator<E> accumulatorFunction);

    @Positive
    public final E accumulateAndGet(int i, E x, BinaryOperator<E> accumulatorFunction);

    @Positive
    public String toString();

    @Positive
    public final E getPlain(int i);

    @Positive
    public final void setPlain(int i, E newValue);

    @Positive
    public final E getOpaque(int i);

    @Positive
    public final void setOpaque(int i, E newValue);

    @Positive
    public final E getAcquire(int i);

    @Positive
    public final void setRelease(int i, E newValue);

    @Positive
    public final E compareAndExchange(int i, E expectedValue, E newValue);

    @Positive
    public final E compareAndExchangeAcquire(int i, E expectedValue, E newValue);

    @Positive
    public final E compareAndExchangeRelease(int i, E expectedValue, E newValue);

    @Positive
    public final boolean weakCompareAndSetVolatile(int i, E expectedValue, E newValue);

    @Positive
    public final boolean weakCompareAndSetAcquire(int i, E expectedValue, E newValue);

    @Positive
    public final boolean weakCompareAndSetRelease(int i, E expectedValue, E newValue);
    @Positive
}
