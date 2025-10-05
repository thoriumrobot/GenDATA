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
package javax.swing.tree;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.EventListener;
import java.util.Hashtable;
import java.util.List;
import java.util.Vector;
import javax.swing.DefaultListSelectionModel;
import javax.swing.event.EventListenerList;
import javax.swing.event.SwingPropertyChangeSupport;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;

@AnnotatedFor({ "interning" })
@SuppressWarnings("serial")
public class DefaultTreeSelectionModel implements Cloneable, Serializable, TreeSelectionModel {

    @Interned
    public static final String SELECTION_MODE_PROPERTY;

    protected SwingPropertyChangeSupport changeSupport;

    protected TreePath[] selection;

    protected EventListenerList listenerList;

    protected transient RowMapper rowMapper;

    protected DefaultListSelectionModel listSelectionModel;

    protected int selectionMode;

    protected TreePath leadPath;

    protected int leadIndex;

    protected int leadRow;

    public DefaultTreeSelectionModel() {
    }

    public void setRowMapper(RowMapper newMapper);

    public RowMapper getRowMapper();

    public void setSelectionMode(int mode);

    public int getSelectionMode();

    public void setSelectionPath(TreePath path);

    public void setSelectionPaths(TreePath[] pPaths);

    public void addSelectionPath(TreePath path);

    public void addSelectionPaths(TreePath[] paths);

    public void removeSelectionPath(TreePath path);

    public void removeSelectionPaths(TreePath[] paths);

    public TreePath getSelectionPath();

    public TreePath[] getSelectionPaths();

    public int getSelectionCount();

    public boolean isPathSelected(TreePath path);

    public boolean isSelectionEmpty();

    public void clearSelection();

    public void addTreeSelectionListener(TreeSelectionListener x);

    public void removeTreeSelectionListener(TreeSelectionListener x);

    public TreeSelectionListener[] getTreeSelectionListeners();

    protected void fireValueChanged(TreeSelectionEvent e);

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    public int[] getSelectionRows();

    public int getMinSelectionRow();

    public int getMaxSelectionRow();

    public boolean isRowSelected(int row);

    public void resetRowSelection();

    public int getLeadSelectionRow();

    public TreePath getLeadSelectionPath();

    public synchronized void addPropertyChangeListener(PropertyChangeListener listener);

    public synchronized void removePropertyChangeListener(PropertyChangeListener listener);

    public PropertyChangeListener[] getPropertyChangeListeners();

    protected void insureRowContinuity();

    protected boolean arePathsContiguous(TreePath[] paths);

    protected boolean canPathsBeAdded(TreePath[] paths);

    protected boolean canPathsBeRemoved(TreePath[] paths);

    @Deprecated
    protected void notifyPathChange(Vector<?> changedPaths, TreePath oldLeadSelection);

    protected void updateLeadIndex();

    protected void insureUniqueness();

    public String toString();

    public Object clone() throws CloneNotSupportedException;
}

class PathPlaceHolder {

    protected boolean isNew;

    protected TreePath path;
}
