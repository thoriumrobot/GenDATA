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
import java.lang.invoke.VarHandle;
    @Positive
import java.util.function.IntBinaryOperator;
    @Positive
import java.util.function.IntUnaryOperator;
    @Positive
import jdk.internal.misc.Unsafe;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AtomicInteger extends Number implements java.io.Serializable {

    @Positive
    public AtomicInteger(int initialValue) {
    @Positive
    }

    @Positive
    public AtomicInteger() {
    @Positive
    }

    @Positive
    public final int get();

    @Positive
    public final void set(int newValue);

    @Positive
    public final void lazySet(int newValue);

    @Positive
    public final int getAndSet(int newValue);

    @Positive
    public final boolean compareAndSet(int expectedValue, int newValue);

    @Positive
    @Deprecated()
    @Positive
    public final boolean weakCompareAndSet(int expectedValue, int newValue);

    @Positive
    public final boolean weakCompareAndSetPlain(int expectedValue, int newValue);

    @Positive
    public final int getAndIncrement();

    @Positive
    public final int getAndDecrement();

    @Positive
    public final int getAndAdd(int delta);

    @Positive
    public final int incrementAndGet();

    @Positive
    public final int decrementAndGet();

    @Positive
    public final int addAndGet(int delta);

    @Positive
    public final int getAndUpdate(IntUnaryOperator updateFunction);

    @Positive
    public final int updateAndGet(IntUnaryOperator updateFunction);

    @Positive
    public final int getAndAccumulate(int x, IntBinaryOperator accumulatorFunction);

    @Positive
    public final int accumulateAndGet(int x, IntBinaryOperator accumulatorFunction);

    @Positive
    public String toString();

    @Positive
    public int intValue();

    @Positive
    public long longValue();

    @Positive
    public float floatValue();

    @Positive
    public double doubleValue();

    @Positive
    public final int getPlain();

    @Positive
    public final void setPlain(int newValue);

    @Positive
    public final int getOpaque();

    @Positive
    public final void setOpaque(int newValue);

    @Positive
    public final int getAcquire();

    @Positive
    public final void setRelease(int newValue);

    @Positive
    public final int compareAndExchange(int expectedValue, int newValue);

    @Positive
    public final int compareAndExchangeAcquire(int expectedValue, int newValue);

    @Positive
    public final int compareAndExchangeRelease(int expectedValue, int newValue);

    @Positive
    public final boolean weakCompareAndSetVolatile(int expectedValue, int newValue);

    @Positive
    public final boolean weakCompareAndSetAcquire(int expectedValue, int newValue);

    @Positive
    public final boolean weakCompareAndSetRelease(int expectedValue, int newValue);
    @Positive
}

// CFWR semantic augmentation - variant 1
