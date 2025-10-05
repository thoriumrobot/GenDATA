/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2006, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.jgss;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.ietf.jgss.MessageProp;
    @Positive
import java.util.LinkedList;

    @Positive
public class TokenTracker {

    @Positive
    public TokenTracker(int initNumber) {
    @Positive
    }

    @Positive
    synchronized public final void getProps(int number, MessageProp prop);

    @Positive
    public String toString();

    @Positive
    class Entry {

    @Positive
        final int compareTo(int number);

    @Positive
        @Pure
    @Positive
        final boolean contains(int number);

    @Positive
        final void append(int number);

    @Positive
        final void setInterval(int start, int end);

    @Positive
        final void setEnd(int end);

    @Positive
        final void setStart(int start);

    @Positive
        final int getStart();

    @Positive
        final int getEnd();

    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
