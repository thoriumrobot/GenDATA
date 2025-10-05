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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import java.util.function.BinaryOperator;
    @Positive
import java.util.function.UnaryOperator;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AtomicReference<V> implements java.io.Serializable {

    @Positive
    public AtomicReference(V initialValue) {
    @Positive
    }

    @Positive
    public AtomicReference() {
    @Positive
    }

    @Positive
    public final V get();

    @Positive
    public final void set(V newValue);

    @Positive
    public final void lazySet(V newValue);

    @Positive
    public final boolean compareAndSet(V expectedValue, V newValue);

    @Positive
    @Deprecated()
    @Positive
    public final boolean weakCompareAndSet(V expectedValue, V newValue);

    @Positive
    public final boolean weakCompareAndSetPlain(V expectedValue, V newValue);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public final V getAndSet(V newValue);

    @Positive
    public final V getAndUpdate(UnaryOperator<V> updateFunction);

    @Positive
    public final V updateAndGet(UnaryOperator<V> updateFunction);

    @Positive
    public final V getAndAccumulate(V x, BinaryOperator<V> accumulatorFunction);

    @Positive
    public final V accumulateAndGet(V x, BinaryOperator<V> accumulatorFunction);

    @Positive
    public String toString();

    @Positive
    public final V getPlain();

    @Positive
    public final void setPlain(V newValue);

    @Positive
    public final V getOpaque();

    @Positive
    public final void setOpaque(V newValue);

    @Positive
    public final V getAcquire();

    @Positive
    public final void setRelease(V newValue);

    @Positive
    public final V compareAndExchange(V expectedValue, V newValue);

    @Positive
    public final V compareAndExchangeAcquire(V expectedValue, V newValue);

    @Positive
    public final V compareAndExchangeRelease(V expectedValue, V newValue);

    @Positive
    public final boolean weakCompareAndSetVolatile(V expectedValue, V newValue);

    @Positive
    public final boolean weakCompareAndSetAcquire(V expectedValue, V newValue);

    @Positive
    public final boolean weakCompareAndSetRelease(V expectedValue, V newValue);
    @Positive
}
