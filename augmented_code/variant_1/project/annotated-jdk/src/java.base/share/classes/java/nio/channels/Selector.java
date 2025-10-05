/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels;

    @Positive
import org.checkerframework.checker.calledmethods.qual.EnsuresCalledMethodsIf;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.Closeable;
    @Positive
import java.io.IOException;
    @Positive
import java.nio.channels.spi.SelectorProvider;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Consumer;

    @Positive
@AnnotatedFor({ "interning", "mustcall" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Selector implements Closeable {

    @Positive
    protected Selector() {
    @Positive
    }

    @Positive
    public static Selector open() throws IOException;

    @Positive
    @EnsuresCalledMethodsIf(expression = "this", result = false, methods = { "close" })
    @Positive
    public abstract boolean isOpen();

    @Positive
    public abstract SelectorProvider provider();

    @Positive
    public abstract Set<SelectionKey> keys();

    @Positive
    public abstract Set<SelectionKey> selectedKeys();

    @Positive
    public abstract int selectNow() throws IOException;

    @Positive
    public abstract int select(long timeout) throws IOException;

    @Positive
    public abstract int select() throws IOException;

    @Positive
    public int select(Consumer<SelectionKey> action, long timeout) throws IOException;

    @Positive
    public int select(Consumer<SelectionKey> action) throws IOException;

    @Positive
    public int selectNow(Consumer<SelectionKey> action) throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public abstract Selector wakeup(@MustCallAlias Selector this);

    @Positive
    public abstract void close() throws IOException;
    @Positive
}
