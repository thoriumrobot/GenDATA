/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2017, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.datatransfer;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import sun.datatransfer.DataFlavorUtil;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Clipboard {

    @Positive
    protected ClipboardOwner owner;

    @Positive
    protected Transferable contents;

    @Positive
    public Clipboard(String name) {
    @Positive
    }

    @Positive
    public String getName();

    @Positive
    public synchronized void setContents(Transferable contents, ClipboardOwner owner);

    @Positive
    public synchronized Transferable getContents(Object requestor);

    @Positive
    public DataFlavor[] getAvailableDataFlavors();

    @Positive
    public boolean isDataFlavorAvailable(DataFlavor flavor);

    @Positive
    public Object getData(DataFlavor flavor) throws UnsupportedFlavorException, IOException;

    @Positive
    public synchronized void addFlavorListener(FlavorListener listener);

    @Positive
    public synchronized void removeFlavorListener(FlavorListener listener);

    @Positive
    public synchronized FlavorListener[] getFlavorListeners();
    @Positive
}
