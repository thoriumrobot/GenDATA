/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.tree;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.BitSet;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.List;
    @Positive
import java.util.Vector;
    @Positive
import javax.swing.DefaultListSelectionModel;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import javax.swing.event.SwingPropertyChangeSupport;
    @Positive
import javax.swing.event.TreeSelectionEvent;
    @Positive
import javax.swing.event.TreeSelectionListener;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class DefaultTreeSelectionModel implements Cloneable, Serializable, TreeSelectionModel {

    @Positive
    @Interned
    @Positive
    public static final String SELECTION_MODE_PROPERTY;

    @Positive
    protected SwingPropertyChangeSupport changeSupport;

    @Positive
    protected TreePath[] selection;

    @Positive
    protected EventListenerList listenerList;

    @Positive
    protected transient RowMapper rowMapper;

    @Positive
    protected DefaultListSelectionModel listSelectionModel;

    @Positive
    protected int selectionMode;

    @Positive
    protected TreePath leadPath;

    @Positive
    protected int leadIndex;

    @Positive
    protected int leadRow;

    @Positive
    public DefaultTreeSelectionModel() {
    @Positive
    }

    @Positive
    public void setRowMapper(RowMapper newMapper);

    @Positive
    public RowMapper getRowMapper();

    @Positive
    public void setSelectionMode(int mode);

    @Positive
    public int getSelectionMode();

    @Positive
    public void setSelectionPath(TreePath path);

    @Positive
    public void setSelectionPaths(TreePath[] pPaths);

    @Positive
    public void addSelectionPath(TreePath path);

    @Positive
    public void addSelectionPaths(TreePath[] paths);

    @Positive
    public void removeSelectionPath(TreePath path);

    @Positive
    public void removeSelectionPaths(TreePath[] paths);

    @Positive
    public TreePath getSelectionPath();

    @Positive
    public TreePath[] getSelectionPaths();

    @Positive
    public int getSelectionCount();

    @Positive
    public boolean isPathSelected(TreePath path);

    @Positive
    public boolean isSelectionEmpty();

    @Positive
    public void clearSelection();

    @Positive
    public void addTreeSelectionListener(TreeSelectionListener x);

    @Positive
    public void removeTreeSelectionListener(TreeSelectionListener x);

    @Positive
    public TreeSelectionListener[] getTreeSelectionListeners();

    @Positive
    protected void fireValueChanged(TreeSelectionEvent e);

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    public int[] getSelectionRows();

    @Positive
    public int getMinSelectionRow();

    @Positive
    public int getMaxSelectionRow();

    @Positive
    public boolean isRowSelected(int row);

    @Positive
    public void resetRowSelection();

    @Positive
    public int getLeadSelectionRow();

    @Positive
    public TreePath getLeadSelectionPath();

    @Positive
    public synchronized void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public synchronized void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public PropertyChangeListener[] getPropertyChangeListeners();

    @Positive
    protected void insureRowContinuity();

    @Positive
    protected boolean arePathsContiguous(TreePath[] paths);

    @Positive
    protected boolean canPathsBeAdded(TreePath[] paths);

    @Positive
    protected boolean canPathsBeRemoved(TreePath[] paths);

    @Positive
    @Deprecated
    @Positive
    protected void notifyPathChange(Vector<?> changedPaths, TreePath oldLeadSelection);

    @Positive
    protected void updateLeadIndex();

    @Positive
    protected void insureUniqueness();

    @Positive
    public String toString();

    @Positive
    public Object clone() throws CloneNotSupportedException;
    @Positive
}

    @Positive
class PathPlaceHolder {

    @Positive
    protected boolean isNew;

    @Positive
    protected TreePath path;
    @Positive
}

// CFWR semantic augmentation - variant 1
