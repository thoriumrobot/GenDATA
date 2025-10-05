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
import java.util.function.LongBinaryOperator;
    @Positive
import java.util.function.LongUnaryOperator;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AtomicLongArray implements java.io.Serializable {

    @Positive
    public AtomicLongArray(int length) {
    @Positive
    }

    @Positive
    public AtomicLongArray(long[] array) {
    @Positive
    }

    @Positive
    public final int length();

    @Positive
    public final long get(int i);

    @Positive
    public final void set(int i, long newValue);

    @Positive
    public final void lazySet(int i, long newValue);

    @Positive
    public final long getAndSet(int i, long newValue);

    @Positive
    public final boolean compareAndSet(int i, long expectedValue, long newValue);

    @Positive
    @Deprecated()
    @Positive
    public final boolean weakCompareAndSet(int i, long expectedValue, long newValue);

    @Positive
    public final boolean weakCompareAndSetPlain(int i, long expectedValue, long newValue);

    @Positive
    public final long getAndIncrement(int i);

    @Positive
    public final long getAndDecrement(int i);

    @Positive
    public final long getAndAdd(int i, long delta);

    @Positive
    public final long incrementAndGet(int i);

    @Positive
    public final long decrementAndGet(int i);

    @Positive
    public long addAndGet(int i, long delta);

    @Positive
    public final long getAndUpdate(int i, LongUnaryOperator updateFunction);

    @Positive
    public final long updateAndGet(int i, LongUnaryOperator updateFunction);

    @Positive
    public final long getAndAccumulate(int i, long x, LongBinaryOperator accumulatorFunction);

    @Positive
    public final long accumulateAndGet(int i, long x, LongBinaryOperator accumulatorFunction);

    @Positive
    public String toString();

    @Positive
    public final long getPlain(int i);

    @Positive
    public final void setPlain(int i, long newValue);

    @Positive
    public final long getOpaque(int i);

    @Positive
    public final void setOpaque(int i, long newValue);

    @Positive
    public final long getAcquire(int i);

    @Positive
    public final void setRelease(int i, long newValue);

    @Positive
    public final long compareAndExchange(int i, long expectedValue, long newValue);

    @Positive
    public final long compareAndExchangeAcquire(int i, long expectedValue, long newValue);

    @Positive
    public final long compareAndExchangeRelease(int i, long expectedValue, long newValue);

    @Positive
    public final boolean weakCompareAndSetVolatile(int i, long expectedValue, long newValue);

    @Positive
    public final boolean weakCompareAndSetAcquire(int i, long expectedValue, long newValue);

    @Positive
    public final boolean weakCompareAndSetRelease(int i, long expectedValue, long newValue);
    @Positive
}
