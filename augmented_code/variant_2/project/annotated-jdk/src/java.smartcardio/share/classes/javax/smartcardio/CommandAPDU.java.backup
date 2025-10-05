/*
 * Copyright (c) 2005, 2006, Oracle and/or its affiliates. All rights reserved.
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
package javax.smartcardio;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Arrays;
import java.nio.ByteBuffer;

public final class CommandAPDU implements java.io.Serializable {

    public CommandAPDU(byte[] apdu) {
    }

    public CommandAPDU(byte[] apdu, int apduOffset, int apduLength) {
    }

    public CommandAPDU(ByteBuffer apdu) {
    }

    public CommandAPDU(int cla, int ins, int p1, int p2) {
    }

    public CommandAPDU(int cla, int ins, int p1, int p2, int ne) {
    }

    public CommandAPDU(int cla, int ins, int p1, int p2, byte[] data) {
    }

    public CommandAPDU(int cla, int ins, int p1, int p2, byte[] data, int dataOffset, int dataLength) {
    }

    public CommandAPDU(int cla, int ins, int p1, int p2, byte[] data, int ne) {
    }

    public CommandAPDU(int cla, int ins, int p1, int p2, byte[] data, int dataOffset, int dataLength, int ne) {
    }

    public int getCLA();

    public int getINS();

    public int getP1();

    public int getP2();

    public int getNc();

    public byte[] getData();

    public int getNe();

    public byte[] getBytes();

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();
}
