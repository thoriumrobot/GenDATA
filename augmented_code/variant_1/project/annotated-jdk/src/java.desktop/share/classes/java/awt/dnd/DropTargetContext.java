/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.dnd;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.datatransfer.DataFlavor;
    @Positive
import java.awt.datatransfer.Transferable;
    @Positive
import java.awt.datatransfer.UnsupportedFlavorException;
    @Positive
import java.awt.dnd.peer.DropTargetContextPeer;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.List;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AWTAccessor.DropTargetContextAccessor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class DropTargetContext implements Serializable {

    @Positive
    public DropTarget getDropTarget();

    @Positive
    public Component getComponent();

    @Positive
    void reset();

    @Positive
    protected void setTargetActions(int actions);

    @Positive
    protected int getTargetActions();

    @Positive
    public void dropComplete(boolean success) throws InvalidDnDOperationException;

    @Positive
    protected void acceptDrag(int dragOperation);

    @Positive
    protected void rejectDrag();

    @Positive
    protected void acceptDrop(int dropOperation);

    @Positive
    protected void rejectDrop();

    @Positive
    protected DataFlavor[] getCurrentDataFlavors();

    @Positive
    protected List<DataFlavor> getCurrentDataFlavorsAsList();

    @Positive
    protected boolean isDataFlavorSupported(DataFlavor df);

    @Positive
    protected Transferable getTransferable() throws InvalidDnDOperationException;

    @Positive
    DropTargetContextPeer getDropTargetContextPeer();

    @Positive
    void setDropTargetContextPeer(final DropTargetContextPeer dtcp);

    @Positive
    protected Transferable createTransferableProxy(Transferable t, boolean local);

    @Positive
    protected class TransferableProxy implements Transferable {

    @Positive
        public DataFlavor[] getTransferDataFlavors();

    @Positive
        public boolean isDataFlavorSupported(DataFlavor flavor);

    @Positive
        public Object getTransferData(DataFlavor df) throws UnsupportedFlavorException, IOException;

    @Positive
        protected Transferable transferable;

    @Positive
        protected boolean isLocal;
    @Positive
    }
    @Positive
}
