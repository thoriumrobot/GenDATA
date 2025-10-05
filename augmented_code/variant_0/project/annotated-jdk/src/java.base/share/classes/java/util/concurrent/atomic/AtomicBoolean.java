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
public class AtomicBoolean implements java.io.Serializable {

    @Positive
    public AtomicBoolean(boolean initialValue) {
    @Positive
    }

    @Positive
    public AtomicBoolean() {
    @Positive
    }

    @Positive
    public final boolean get();

    @Positive
    public final boolean compareAndSet(boolean expectedValue, boolean newValue);

    @Positive
    @Deprecated()
    @Positive
    public boolean weakCompareAndSet(boolean expectedValue, boolean newValue);

    @Positive
    public boolean weakCompareAndSetPlain(boolean expectedValue, boolean newValue);

    @Positive
    public final void set(boolean newValue);

    @Positive
    public final void lazySet(boolean newValue);

    @Positive
    public final boolean getAndSet(boolean newValue);

    @Positive
    public String toString();

    @Positive
    public final boolean getPlain();

    @Positive
    public final void setPlain(boolean newValue);

    @Positive
    public final boolean getOpaque();

    @Positive
    public final void setOpaque(boolean newValue);

    @Positive
    public final boolean getAcquire();

    @Positive
    public final void setRelease(boolean newValue);

    @Positive
    public final boolean compareAndExchange(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean compareAndExchangeAcquire(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean compareAndExchangeRelease(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean weakCompareAndSetVolatile(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean weakCompareAndSetAcquire(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean weakCompareAndSetRelease(boolean expectedValue, boolean newValue);
    @Positive
}
