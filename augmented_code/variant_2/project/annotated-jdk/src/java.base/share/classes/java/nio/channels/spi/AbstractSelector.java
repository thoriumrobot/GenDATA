/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels.spi;

    @Positive
import org.checkerframework.checker.calledmethods.qual.EnsuresCalledMethodsIf;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.nio.channels.SelectionKey;
    @Positive
import java.nio.channels.Selector;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Set;
    @Positive
import sun.nio.ch.Interruptible;
    @Positive
import sun.nio.ch.SelectorImpl;

    @Positive
@AnnotatedFor({ "mustcall" })
    @Positive
public abstract class AbstractSelector extends Selector {

    @Positive
    protected AbstractSelector(SelectorProvider provider) {
    @Positive
    }

    @Positive
    void cancel(SelectionKey k);

    @Positive
    public final void close() throws IOException;

    @Positive
    protected abstract void implCloseSelector() throws IOException;

    @Positive
    @EnsuresCalledMethodsIf(expression = "this", result = false, methods = { "close" })
    @Positive
    public final boolean isOpen();

    @Positive
    public final SelectorProvider provider();

    @Positive
    protected final Set<SelectionKey> cancelledKeys();

    @Positive
    protected abstract SelectionKey register(AbstractSelectableChannel ch, int ops, Object att);

    @Positive
    protected final void deregister(AbstractSelectionKey key);

    @Positive
    protected final void begin();

    @Positive
    protected final void end();
    @Positive
}
