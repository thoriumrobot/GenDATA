/*
 * Copyright (c) 1997, 2015, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.table;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.framework.qual.AnnotatedFor;
import javax.swing.*;
import javax.swing.event.*;
import java.io.Serializable;
import java.util.EventListener;

@AnnotatedFor({ "index" })
@SuppressWarnings("serial")
public abstract class AbstractTableModel implements TableModel, Serializable {

    protected EventListenerList listenerList;

    protected AbstractTableModel() {
    }

    public String getColumnName(@NonNegative int column);

    @GTENegativeOne
    public int findColumn(String columnName);

    public Class<?> getColumnClass(@NonNegative int columnIndex);

    public boolean isCellEditable(@NonNegative int rowIndex, @NonNegative int columnIndex);

    public void setValueAt(Object aValue, @NonNegative int rowIndex, @NonNegative int columnIndex);

    public void addTableModelListener(TableModelListener l);

    public void removeTableModelListener(TableModelListener l);

    public TableModelListener[] getTableModelListeners();

    public void fireTableDataChanged();

    public void fireTableStructureChanged();

    public void fireTableRowsInserted(@NonNegative int firstRow, @NonNegative int lastRow);

    public void fireTableRowsUpdated(@NonNegative int firstRow, @NonNegative int lastRow);

    public void fireTableRowsDeleted(@NonNegative int firstRow, @NonNegative int lastRow);

    public void fireTableCellUpdated(@NonNegative int row, @NonNegative int column);

    public void fireTableChanged(TableModelEvent e);

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);
}
