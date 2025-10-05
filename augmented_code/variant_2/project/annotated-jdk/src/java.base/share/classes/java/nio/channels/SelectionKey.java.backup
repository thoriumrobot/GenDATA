/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.mustcall.qual.NotOwning;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

@AnnotatedFor({ "interning", "mustcall" })
@UsesObjectEquals
public abstract class SelectionKey {

    protected SelectionKey() {
    }

    @NotOwning
    public abstract SelectableChannel channel();

    public abstract Selector selector();

    public abstract boolean isValid();

    public abstract void cancel();

    public abstract int interestOps();

    public abstract SelectionKey interestOps(int ops);

    public int interestOpsOr(int ops);

    public int interestOpsAnd(int ops);

    public abstract int readyOps();

    public static final int OP_READ;

    public static final int OP_WRITE;

    public static final int OP_CONNECT;

    public static final int OP_ACCEPT;

    public final boolean isReadable();

    public final boolean isWritable();

    public final boolean isConnectable();

    public final boolean isAcceptable();

    public final Object attach(Object ob);

    public final Object attachment();
}
