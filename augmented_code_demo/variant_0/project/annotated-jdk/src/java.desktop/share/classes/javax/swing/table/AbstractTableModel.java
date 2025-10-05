/*
    @Positive
 * Copyright (c) 1997, 2015, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.table;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.event.*;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.EventListener;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public abstract class AbstractTableModel implements TableModel, Serializable {

    @Positive
    protected EventListenerList listenerList;

    @Positive
    protected AbstractTableModel() {
    @Positive
    }

    @Positive
    public String getColumnName(@NonNegative int column);

    @Positive
    @GTENegativeOne
    @Positive
    public int findColumn(String columnName);

    @Positive
    public Class<?> getColumnClass(@NonNegative int columnIndex);

    @Positive
    public boolean isCellEditable(@NonNegative int rowIndex, @NonNegative int columnIndex);

    @Positive
    public void setValueAt(Object aValue, @NonNegative int rowIndex, @NonNegative int columnIndex);

    @Positive
    public void addTableModelListener(TableModelListener l);

    @Positive
    public void removeTableModelListener(TableModelListener l);

    @Positive
    public TableModelListener[] getTableModelListeners();

    @Positive
    public void fireTableDataChanged();

    @Positive
    public void fireTableStructureChanged();

    @Positive
    public void fireTableRowsInserted(@NonNegative int firstRow, @NonNegative int lastRow);

    @Positive
    public void fireTableRowsUpdated(@NonNegative int firstRow, @NonNegative int lastRow);

    @Positive
    public void fireTableRowsDeleted(@NonNegative int firstRow, @NonNegative int lastRow);

    @Positive
    public void fireTableCellUpdated(@NonNegative int row, @NonNegative int column);

    @Positive
    public void fireTableChanged(TableModelEvent e);

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);
    @Positive
}

// CFWR semantic augmentation - variant 0
