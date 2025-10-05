/*
    @Positive
 * Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.
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

// CFWR semantic augmentation - variant 0
