/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.dnd;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Component;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.dnd.peer.DropTargetContextPeer;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import sun.awt.AWTAccessor;
import sun.awt.AWTAccessor.DropTargetContextAccessor;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class DropTargetContext implements Serializable {

    public DropTarget getDropTarget();

    public Component getComponent();

    void reset();

    protected void setTargetActions(int actions);

    protected int getTargetActions();

    public void dropComplete(boolean success) throws InvalidDnDOperationException;

    protected void acceptDrag(int dragOperation);

    protected void rejectDrag();

    protected void acceptDrop(int dropOperation);

    protected void rejectDrop();

    protected DataFlavor[] getCurrentDataFlavors();

    protected List<DataFlavor> getCurrentDataFlavorsAsList();

    protected boolean isDataFlavorSupported(DataFlavor df);

    protected Transferable getTransferable() throws InvalidDnDOperationException;

    DropTargetContextPeer getDropTargetContextPeer();

    void setDropTargetContextPeer(final DropTargetContextPeer dtcp);

    protected Transferable createTransferableProxy(Transferable t, boolean local);

    protected class TransferableProxy implements Transferable {

        public DataFlavor[] getTransferDataFlavors();

        public boolean isDataFlavorSupported(DataFlavor flavor);

        public Object getTransferData(DataFlavor df) throws UnsupportedFlavorException, IOException;

        protected Transferable transferable;

        protected boolean isLocal;
    }
}
